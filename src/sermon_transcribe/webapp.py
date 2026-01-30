import json
import os
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit

from sermon_transcribe.cleanup import (
    apply_disclaimer,
    build_prompt,
    build_summary_chunk_prompt,
    build_summary_merge_prompt,
    build_summary_prompt,
    call_claude,
    split_into_chunks,
)
from sermon_transcribe.io_utils import ensure_dir
from sermon_transcribe.transcription import (
    TranscriptionResult,
    build_config,
    build_model,
    output_paths,
    transcribe_file,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4GB max file size
app.config["UPLOAD_FOLDER"] = Path("/app/uploads").resolve()
app.config["OUTPUT_FOLDER"] = Path("/app/output").resolve()
app.config["MODEL_CACHE"] = Path("/app/model_cache").resolve()

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=4 * 1024 * 1024 * 1024,  # 4GB max for uploads
    async_mode='threading'  # Use threading instead of eventlet
)

# Global state for jobs
jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()

# Processing semaphore - ensures only one transcription runs at a time
processing_semaphore = threading.Semaphore(1)

# Model instance (loaded once at startup)
model = None
config = None


def init_model():
    """Initialize the Whisper model at startup."""
    global model, config
    ensure_dir(app.config["MODEL_CACHE"])

    model_name = os.environ.get("WHISPER_MODEL", "large-v3")
    device = os.environ.get("WHISPER_DEVICE", "auto")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE")
    hf_token = os.environ.get("HF_TOKEN")

    config = build_config(
        model=model_name,
        device=device,
        compute_type=compute_type,
        beam_size=5,
        language=None,
        task="transcribe",
        vad_filter=True,
        cache_dir=app.config["MODEL_CACHE"],
        hf_token=hf_token,
    )

    print(f"Loading model: {config.model}", flush=True)
    model = build_model(config)
    print("Model loaded successfully", flush=True)


def emit_progress(job_id: str, status: str, progress: int, message: str, data: Optional[dict] = None):
    """Emit progress update via WebSocket."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status
            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"] = message
            jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
            if data:
                jobs[job_id].update(data)

    payload = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
    }
    if data:
        payload.update(data)

    socketio.emit("progress", payload, namespace="/")


def convert_audio(job_id: str, audio_path: Path, output_dir: Path, convert_m4a: bool, convert_flac: bool):
    """Convert audio to M4A and/or FLAC with mono and -20 LUFS normalization."""
    results = {}
    base_name = audio_path.stem

    try:
        if convert_m4a:
            emit_progress(job_id, "processing", 51, "Converting to M4A (analyzing audio)...")
            m4a_path = output_dir / f"{base_name}_mono_normalized.m4a"

            # Use two-pass loudnorm filter for accurate -20 LUFS normalization
            # First pass: analyze audio
            cmd_analyze = [
                "ffmpeg", "-i", str(audio_path),
                "-af", "loudnorm=I=-20:TP=-1.5:LRA=11:print_format=json",
                "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd_analyze,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Extract loudnorm parameters from analysis
            loudnorm_stats = None
            output_lines = result.stderr.split('\n')
            json_start = -1
            for i, line in enumerate(output_lines):
                if '"input_i"' in line:
                    json_start = i - 1
                    break

            if json_start >= 0:
                json_str = '\n'.join(output_lines[json_start:])
                try:
                    loudnorm_stats = json.loads(json_str)
                except:
                    pass

            # Second pass: apply normalization with analyzed parameters
            emit_progress(job_id, "processing", 53, "Converting to M4A (normalizing and encoding)...")

            if loudnorm_stats:
                cmd_convert = [
                    "ffmpeg", "-i", str(audio_path),
                    "-af",
                    f"loudnorm=I=-20:TP=-1.5:LRA=11:"
                    f"measured_I={loudnorm_stats['input_i']}:"
                    f"measured_TP={loudnorm_stats['input_tp']}:"
                    f"measured_LRA={loudnorm_stats['input_lra']}:"
                    f"measured_thresh={loudnorm_stats['input_thresh']}:"
                    f"offset={loudnorm_stats['target_offset']}:"
                    f"linear=true:print_format=summary",
                    "-ac", "1",  # Convert to mono
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-y",
                    str(m4a_path)
                ]
            else:
                # Fallback to single-pass if analysis failed
                cmd_convert = [
                    "ffmpeg", "-i", str(audio_path),
                    "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
                    "-ac", "1",  # Convert to mono
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-y",
                    str(m4a_path)
                ]

            subprocess.run(cmd_convert, check=True, capture_output=True, timeout=600)
            results["m4a_path"] = str(m4a_path.absolute())
            results["m4a_url"] = f"/download/{job_id}/m4a"
            emit_progress(job_id, "processing", 55, "M4A conversion complete")

        if convert_flac:
            emit_progress(job_id, "processing", 56, "Converting to FLAC (analyzing audio)...")
            flac_path = output_dir / f"{base_name}_mono_normalized.flac"

            # Similar two-pass process for FLAC
            cmd_analyze = [
                "ffmpeg", "-i", str(audio_path),
                "-af", "loudnorm=I=-20:TP=-1.5:LRA=11:print_format=json",
                "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd_analyze,
                capture_output=True,
                text=True,
                timeout=300
            )

            loudnorm_stats = None
            output_lines = result.stderr.split('\n')
            json_start = -1
            for i, line in enumerate(output_lines):
                if '"input_i"' in line:
                    json_start = i - 1
                    break

            if json_start >= 0:
                json_str = '\n'.join(output_lines[json_start:])
                try:
                    loudnorm_stats = json.loads(json_str)
                except:
                    pass

            emit_progress(job_id, "processing", 58, "Converting to FLAC (normalizing and encoding)...")

            if loudnorm_stats:
                cmd_convert = [
                    "ffmpeg", "-i", str(audio_path),
                    "-af",
                    f"loudnorm=I=-20:TP=-1.5:LRA=11:"
                    f"measured_I={loudnorm_stats['input_i']}:"
                    f"measured_TP={loudnorm_stats['input_tp']}:"
                    f"measured_LRA={loudnorm_stats['input_lra']}:"
                    f"measured_thresh={loudnorm_stats['input_thresh']}:"
                    f"offset={loudnorm_stats['target_offset']}:"
                    f"linear=true:print_format=summary",
                    "-ac", "1",  # Convert to mono
                    "-c:a", "flac",
                    "-y",
                    str(flac_path)
                ]
            else:
                cmd_convert = [
                    "ffmpeg", "-i", str(audio_path),
                    "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
                    "-ac", "1",  # Convert to mono
                    "-c:a", "flac",
                    "-y",
                    str(flac_path)
                ]

            subprocess.run(cmd_convert, check=True, capture_output=True, timeout=600)
            results["flac_path"] = str(flac_path.absolute())
            results["flac_url"] = f"/download/{job_id}/flac"
            emit_progress(job_id, "processing", 60, "FLAC conversion complete")

    except Exception as exc:
        print(f"Audio conversion error for {job_id}: {exc}", flush=True)
        # Don't fail the entire job if conversion fails
        emit_progress(job_id, "processing", 50, f"Audio conversion warning: {str(exc)}")

    return results


def process_transcription(job_id: str, audio_path: Path, output_dir: Path, do_cleanup: bool, convert_m4a: bool = False, convert_flac: bool = False):
    """Background task for transcribing audio."""
    # Small delay to ensure client receives upload response before WebSocket updates
    time.sleep(0.2)

    # Wait for semaphore (ensures sequential processing)
    emit_progress(job_id, "queued", 5, "Waiting in processing queue...")

    with processing_semaphore:
        try:
            emit_progress(job_id, "processing", 10, "Starting transcription...")
            emit_progress(job_id, "processing", 15, "Transcribing audio... (this may take several minutes)")

            result = transcribe_file(
                model=model,
                source_path=audio_path,
                output_dir=output_dir,
                config=config,
                convert_flac=False,
                raw_suffix="_raw" if do_cleanup else "",
            )

            emit_progress(job_id, "processing", 50, "Transcription complete")

            # Set initial paths (cleanup_transcript will update text_path if it runs)
            with jobs_lock:
                jobs[job_id]["text_path"] = str(result.text_path.absolute())
                jobs[job_id]["json_path"] = str(result.json_path.absolute())

            # Perform audio conversion if requested
            conversion_results = {}
            if convert_m4a or convert_flac:
                conversion_results = convert_audio(job_id, audio_path, output_dir, convert_m4a, convert_flac)

            # Add conversion results to job
            with jobs_lock:
                jobs[job_id].update(conversion_results)

            if do_cleanup:
                emit_progress(job_id, "processing", 60, "Cleaning transcript with Claude...")
                cleanup_transcript(job_id, result, output_dir)

            download_urls = {
                "text_url": f"/download/{job_id}/text",
                "json_url": f"/download/{job_id}/json",
            }
            download_urls.update({k: v for k, v in conversion_results.items() if k.endswith("_url")})

            emit_progress(
                job_id,
                "completed",
                100,
                "Processing complete",
                download_urls
            )

        except Exception as exc:
            emit_progress(job_id, "failed", 0, f"Error: {str(exc)}")


def cleanup_transcript(job_id: str, result: TranscriptionResult, output_dir: Path):
    """Clean up transcript with Claude and generate summary."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        emit_progress(job_id, "processing", 60, "Skipping cleanup (no API key)")
        return

    model_name = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    raw_text = result.text_path.read_text(encoding="utf-8").strip()

    if not raw_text:
        emit_progress(job_id, "processing", 90, "Skipping cleanup (empty transcript)")
        return

    # Clean transcript
    emit_progress(job_id, "processing", 65, "Cleaning transcript...")
    chunks = split_into_chunks(raw_text, 8000)
    cleaned_chunks = []

    for idx, chunk in enumerate(chunks, start=1):
        emit_progress(job_id, "processing", 65 + (idx * 10 // len(chunks)), f"Cleaning chunk {idx}/{len(chunks)}")
        prompt = build_prompt(chunk)
        cleaned = call_claude(
            prompt=prompt,
            api_key=api_key,
            model=model_name,
            max_tokens=1200,
            temperature=0.1,
            timeout=120,
        )
        cleaned_chunks.append(cleaned.strip())

    cleaned_text = "\n\n".join(chunk for chunk in cleaned_chunks if chunk)
    base_name = result.audio_path.stem
    cleaned_path = output_dir / f"{base_name}.txt"
    cleaned_path.write_text(apply_disclaimer(cleaned_text), encoding="utf-8")

    # Generate summary
    emit_progress(job_id, "processing", 80, "Generating summary...")

    if len(cleaned_text) <= 16000:
        summary_prompt = build_summary_prompt(cleaned_text)
        summary = call_claude(
            prompt=summary_prompt,
            api_key=api_key,
            model=model_name,
            max_tokens=600,
            temperature=0.1,
            timeout=120,
        )
    else:
        summary_chunks = split_into_chunks(cleaned_text, 16000)
        chunk_notes = []
        for idx, chunk in enumerate(summary_chunks, start=1):
            emit_progress(job_id, "processing", 80 + (idx * 5 // len(summary_chunks)), f"Summary chunk {idx}/{len(summary_chunks)}")
            notes = call_claude(
                prompt=build_summary_chunk_prompt(chunk),
                api_key=api_key,
                model=model_name,
                max_tokens=600,
                temperature=0.1,
                timeout=120,
            )
            chunk_notes.append(notes.strip())

        combined_notes = "\n\n".join(note for note in chunk_notes if note)
        summary = call_claude(
            prompt=build_summary_merge_prompt(combined_notes),
            api_key=api_key,
            model=model_name,
            max_tokens=600,
            temperature=0.1,
            timeout=120,
        )

    summary_path = output_dir / f"{base_name}_summary.txt"
    summary_path.write_text(apply_disclaimer(summary), encoding="utf-8")

    with jobs_lock:
        jobs[job_id]["text_path"] = str(cleaned_path.absolute())
        jobs[job_id]["summary_path"] = str(summary_path.absolute())
        jobs[job_id]["summary_url"] = f"/download/{job_id}/summary"

    emit_progress(job_id, "processing", 90, "Cleanup complete")


@app.route("/")
def index():
    """Main upload page."""
    return render_template("index.html")


@app.route("/review/<job_id>")
def review(job_id: str):
    """Review page for editing transcript."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return "Job not found", 404

    if job.get("status") != "completed":
        return "Job not completed yet", 400

    return render_template("review.html", job=job)


@app.route("/api/transcript/<job_id>", methods=["GET"])
def get_transcript(job_id: str):
    """Get transcript content."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job or "text_path" not in job:
        return jsonify({"error": "Transcript not found"}), 404

    text_path = Path(job["text_path"])
    if not text_path.exists():
        return jsonify({"error": "Transcript file not found"}), 404

    # Check if backup exists (indicates manual edits have been made)
    backup_path = text_path.with_suffix(text_path.suffix + ".backup")
    has_manual_edits = backup_path.exists()

    content = text_path.read_text(encoding="utf-8")

    # If manually edited, also load the original (backup) for comparison
    original_content = None
    if has_manual_edits:
        original_content = backup_path.read_text(encoding="utf-8")

    return jsonify({
        "content": content,
        "original_content": original_content,
        "has_manual_edits": has_manual_edits,
        "filename": text_path.name,
        "last_modified": datetime.fromtimestamp(text_path.stat().st_mtime).isoformat()
    })


@app.route("/api/transcript/<job_id>", methods=["POST"])
def save_transcript(job_id: str):
    """Save edited transcript."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job or "text_path" not in job:
        return jsonify({"error": "Transcript not found"}), 404

    text_path = Path(job["text_path"])
    if not text_path.exists():
        return jsonify({"error": "Transcript file not found"}), 404

    data = request.get_json()
    if not data or "content" not in data:
        return jsonify({"error": "No content provided"}), 400

    # Create backup on first edit
    backup_path = text_path.with_suffix(text_path.suffix + ".backup")
    if not backup_path.exists():
        import shutil
        shutil.copy2(text_path, backup_path)

    # Save new content
    text_path.write_text(data["content"], encoding="utf-8")

    return jsonify({
        "success": True,
        "last_modified": datetime.fromtimestamp(text_path.stat().st_mtime).isoformat(),
        "has_backup": backup_path.exists()
    })


@app.route("/api/transcript/<job_id>/reload", methods=["POST"])
def reload_transcript(job_id: str):
    """Reload transcript from backup."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job or "text_path" not in job:
        return jsonify({"error": "Transcript not found"}), 404

    text_path = Path(job["text_path"])
    backup_path = text_path.with_suffix(text_path.suffix + ".backup")

    if not backup_path.exists():
        return jsonify({"error": "No backup available"}), 404

    # Restore from backup
    import shutil
    shutil.copy2(backup_path, text_path)

    content = text_path.read_text(encoding="utf-8")
    return jsonify({
        "success": True,
        "content": content,
        "last_modified": datetime.fromtimestamp(text_path.stat().st_mtime).isoformat()
    })


@app.route("/api/transcript/<job_id>/regenerate", methods=["POST"])
def regenerate_cleaned_transcript(job_id: str):
    """Regenerate cleaned transcript and summary from current transcript content."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job or "text_path" not in job:
        return jsonify({"error": "Transcript not found"}), 404

    text_path = Path(job["text_path"])
    if not text_path.exists():
        return jsonify({"error": "Transcript file not found"}), 404

    # Check if API key is available
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 400

    # Read current transcript content
    current_text = text_path.read_text(encoding="utf-8").strip()
    if not current_text:
        return jsonify({"error": "Transcript is empty"}), 400

    # Run cleanup in background thread to avoid blocking
    def run_regeneration():
        try:
            model_name = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

            # Clean transcript
            chunks = split_into_chunks(current_text, 8000)
            cleaned_chunks = []

            for idx, chunk in enumerate(chunks, start=1):
                prompt = build_prompt(chunk)
                cleaned = call_claude(
                    prompt=prompt,
                    api_key=api_key,
                    model=model_name,
                    max_tokens=1200,
                    temperature=0.1,
                    timeout=120,
                )
                cleaned_chunks.append(cleaned.strip())

            cleaned_text = "\n\n".join(chunk for chunk in cleaned_chunks if chunk)

            # Save cleaned transcript (overwrite current file)
            text_path.write_text(apply_disclaimer(cleaned_text), encoding="utf-8")

            # Generate summary
            output_dir = text_path.parent
            base_name = text_path.stem

            if len(cleaned_text) <= 16000:
                summary_prompt = build_summary_prompt(cleaned_text)
                summary = call_claude(
                    prompt=summary_prompt,
                    api_key=api_key,
                    model=model_name,
                    max_tokens=600,
                    temperature=0.1,
                    timeout=120,
                )
            else:
                summary_chunks = split_into_chunks(cleaned_text, 16000)
                chunk_notes = []
                for chunk in summary_chunks:
                    notes = call_claude(
                        prompt=build_summary_chunk_prompt(chunk),
                        api_key=api_key,
                        model=model_name,
                        max_tokens=600,
                        temperature=0.1,
                        timeout=120,
                    )
                    chunk_notes.append(notes.strip())

                combined_notes = "\n\n".join(note for note in chunk_notes if note)
                summary = call_claude(
                    prompt=build_summary_merge_prompt(combined_notes),
                    api_key=api_key,
                    model=model_name,
                    max_tokens=600,
                    temperature=0.1,
                    timeout=120,
                )

            summary_path = output_dir / f"{base_name}_summary.txt"
            summary_path.write_text(apply_disclaimer(summary), encoding="utf-8")

            with jobs_lock:
                jobs[job_id]["summary_path"] = str(summary_path.absolute())
                jobs[job_id]["summary_url"] = f"/download/{job_id}/summary"

        except Exception as exc:
            print(f"Error regenerating transcript for {job_id}: {exc}", flush=True)

    thread = threading.Thread(target=run_regeneration, daemon=True)
    thread.start()

    return jsonify({
        "success": True,
        "message": "Regeneration started. This may take a few minutes."
    })


@app.route("/api/audio/<job_id>")
def stream_audio(job_id: str):
    """Stream audio file."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return "Job not found", 404

    # Find audio file in uploads directory
    upload_dir = app.config["UPLOAD_FOLDER"] / job_id
    if not upload_dir.exists():
        return "Audio not found", 404

    # Find WAV file
    wav_files = list(upload_dir.glob("*.WAV")) + list(upload_dir.glob("*.wav"))
    if not wav_files:
        return "Audio file not found", 404

    audio_path = wav_files[0]
    return send_file(str(audio_path), mimetype="audio/wav")


@app.route("/api/segments/<job_id>")
def get_segments(job_id: str):
    """Get transcript segments with timestamps."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Find JSON file in output directory
    output_dir = app.config["OUTPUT_FOLDER"] / job_id
    if not output_dir.exists():
        return jsonify({"error": "Output not found"}), 404

    # Find raw JSON file with segments
    json_files = list(output_dir.glob("*_raw.json"))
    if not json_files:
        return jsonify({"error": "Segments not found"}), 404

    json_path = json_files[0]

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to load segments: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload and start transcription."""
    if "audio" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    do_cleanup = request.form.get("cleanup", "false").lower() == "true"
    convert_m4a = request.form.get("convert_m4a", "false").lower() == "true"
    convert_flac = request.form.get("convert_flac", "false").lower() == "true"

    # Generate job ID
    job_id = secrets.token_hex(16)

    # Save uploaded file
    ensure_dir(app.config["UPLOAD_FOLDER"])
    upload_path = app.config["UPLOAD_FOLDER"] / job_id
    ensure_dir(upload_path)

    audio_filename = Path(file.filename).name
    audio_path = upload_path / audio_filename
    file.save(str(audio_path))

    # Create output directory
    output_dir = app.config["OUTPUT_FOLDER"] / job_id
    ensure_dir(output_dir)

    # Initialize job
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "filename": audio_filename,
            "status": "queued",
            "progress": 0,
            "message": "Job queued",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "cleanup": do_cleanup,
        }

    # Start background processing
    thread = threading.Thread(
        target=process_transcription,
        args=(job_id, audio_path, output_dir, do_cleanup, convert_m4a, convert_flac),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/upload-transcript", methods=["POST"])
def upload_transcript():
    """Handle raw transcript upload and optionally process with Claude."""
    if "transcript" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["transcript"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    do_cleanup = request.form.get("cleanup", "false").lower() == "true"

    # Generate job ID
    job_id = secrets.token_hex(16)

    # Create output directory
    output_dir = app.config["OUTPUT_FOLDER"] / job_id
    ensure_dir(output_dir)

    # Read transcript content
    try:
        transcript_content = file.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        return jsonify({"error": "Invalid text file encoding. Please use UTF-8."}), 400

    if not transcript_content:
        return jsonify({"error": "Transcript file is empty"}), 400

    # Save raw transcript
    transcript_filename = Path(file.filename).stem
    raw_suffix = "_raw" if do_cleanup else ""
    text_path = output_dir / f"{transcript_filename}{raw_suffix}.txt"
    text_path.write_text(transcript_content, encoding="utf-8")

    # Initialize job
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "filename": file.filename,
            "status": "processing" if do_cleanup else "completed",
            "progress": 50 if do_cleanup else 100,
            "message": "Processing transcript..." if do_cleanup else "Transcript uploaded",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "cleanup": do_cleanup,
            "text_path": str(text_path.absolute()),
            "text_url": f"/download/{job_id}/text",
        }

    if do_cleanup:
        # Process transcript with Claude in background
        def process_transcript_cleanup():
            try:
                emit_progress(job_id, "processing", 60, "Cleaning transcript with Claude...")

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    emit_progress(job_id, "completed", 100, "Upload complete (no API key for cleanup)")
                    return

                model_name = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

                # Clean transcript
                emit_progress(job_id, "processing", 65, "Cleaning transcript...")
                chunks = split_into_chunks(transcript_content, 8000)
                cleaned_chunks = []

                for idx, chunk in enumerate(chunks, start=1):
                    emit_progress(job_id, "processing", 65 + (idx * 10 // len(chunks)), f"Cleaning chunk {idx}/{len(chunks)}")
                    prompt = build_prompt(chunk)
                    cleaned = call_claude(
                        prompt=prompt,
                        api_key=api_key,
                        model=model_name,
                        max_tokens=1200,
                        temperature=0.1,
                        timeout=120,
                    )
                    cleaned_chunks.append(cleaned.strip())

                cleaned_text = "\n\n".join(chunk for chunk in cleaned_chunks if chunk)
                cleaned_path = output_dir / f"{transcript_filename}.txt"
                cleaned_path.write_text(apply_disclaimer(cleaned_text), encoding="utf-8")

                # Generate summary
                emit_progress(job_id, "processing", 80, "Generating summary...")

                if len(cleaned_text) <= 16000:
                    summary_prompt = build_summary_prompt(cleaned_text)
                    summary = call_claude(
                        prompt=summary_prompt,
                        api_key=api_key,
                        model=model_name,
                        max_tokens=600,
                        temperature=0.1,
                        timeout=120,
                    )
                else:
                    summary_chunks = split_into_chunks(cleaned_text, 16000)
                    chunk_notes = []
                    for idx, chunk in enumerate(summary_chunks, start=1):
                        emit_progress(job_id, "processing", 80 + (idx * 5 // len(summary_chunks)), f"Summary chunk {idx}/{len(summary_chunks)}")
                        notes = call_claude(
                            prompt=build_summary_chunk_prompt(chunk),
                            api_key=api_key,
                            model=model_name,
                            max_tokens=600,
                            temperature=0.1,
                            timeout=120,
                        )
                        chunk_notes.append(notes.strip())

                    combined_notes = "\n\n".join(note for note in chunk_notes if note)
                    summary = call_claude(
                        prompt=build_summary_merge_prompt(combined_notes),
                        api_key=api_key,
                        model=model_name,
                        max_tokens=600,
                        temperature=0.1,
                        timeout=120,
                    )

                summary_path = output_dir / f"{transcript_filename}_summary.txt"
                summary_path.write_text(apply_disclaimer(summary), encoding="utf-8")

                with jobs_lock:
                    jobs[job_id]["text_path"] = str(cleaned_path.absolute())
                    jobs[job_id]["summary_path"] = str(summary_path.absolute())
                    jobs[job_id]["summary_url"] = f"/download/{job_id}/summary"

                emit_progress(job_id, "completed", 100, "Processing complete", {
                    "text_url": f"/download/{job_id}/text",
                    "summary_url": f"/download/{job_id}/summary",
                })

            except Exception as exc:
                emit_progress(job_id, "failed", 0, f"Error: {str(exc)}")

        thread = threading.Thread(target=process_transcript_cleanup, daemon=True)
        thread.start()

    return jsonify({"job_id": job_id, "status": jobs[job_id]["status"]})


@app.route("/status/<job_id>")
def status(job_id: str):
    """Get job status."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job)


@app.route("/jobs")
def list_jobs():
    """List all jobs."""
    with jobs_lock:
        job_list = list(jobs.values())

    # Sort by creation time, newest first
    job_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return jsonify({"jobs": job_list})


@app.route("/download/<job_id>/<file_type>")
def download(job_id: str, file_type: str):
    """Download transcription results."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if file_type == "text" and "text_path" in job:
        return send_file(job["text_path"], as_attachment=True)
    elif file_type == "json" and "json_path" in job:
        return send_file(job["json_path"], as_attachment=True)
    elif file_type == "summary" and "summary_path" in job:
        return send_file(job["summary_path"], as_attachment=True)
    elif file_type == "m4a" and "m4a_path" in job:
        return send_file(job["m4a_path"], as_attachment=True, mimetype="audio/mp4")
    elif file_type == "flac" and "flac_path" in job:
        return send_file(job["flac_path"], as_attachment=True, mimetype="audio/flac")
    else:
        return jsonify({"error": "File not found"}), 404


@app.route("/job/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    """Delete a job and all its associated files."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

    try:
        # Delete output directory
        output_dir = app.config["OUTPUT_FOLDER"] / job_id
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # Delete upload directory
        upload_dir = app.config["UPLOAD_FOLDER"] / job_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

        # Remove from jobs dictionary
        with jobs_lock:
            del jobs[job_id]

        return jsonify({"success": True, "message": "Job deleted successfully"})
    except Exception as exc:
        return jsonify({"error": f"Failed to delete job: {str(exc)}"}), 500


def load_existing_jobs():
    """Load previously processed jobs from output directory."""
    output_folder = app.config["OUTPUT_FOLDER"]
    if not output_folder.exists():
        return

    loaded_count = 0
    for job_dir in output_folder.iterdir():
        if not job_dir.is_dir():
            continue

        job_id = job_dir.name

        # Find the original filename from any of the output files
        txt_files = list(job_dir.glob("*.txt"))
        json_files = list(job_dir.glob("*.json"))

        if not txt_files and not json_files:
            continue

        # Try to get the base filename
        base_filename = None
        text_path = None
        json_path = None
        summary_path = None

        # Look for audio files
        m4a_path = None
        flac_path = None
        m4a_files = list(job_dir.glob("*_mono_normalized.m4a"))
        flac_files = list(job_dir.glob("*_mono_normalized.flac"))

        if m4a_files:
            m4a_path = m4a_files[0]
        if flac_files:
            flac_path = flac_files[0]

        # Look for files
        for txt_file in txt_files:
            if txt_file.stem.endswith("_summary"):
                summary_path = txt_file
            elif txt_file.stem.endswith("_raw"):
                # Check if cleaned version exists
                cleaned_name = txt_file.stem[:-4] + ".txt"  # Remove _raw
                cleaned_path = txt_file.parent / cleaned_name
                if cleaned_path.exists():
                    text_path = cleaned_path
                    base_filename = txt_file.stem[:-4]
                else:
                    text_path = txt_file
                    base_filename = txt_file.stem[:-4]
            elif not txt_file.stem.endswith("_summary"):
                text_path = txt_file
                if base_filename is None:
                    base_filename = txt_file.stem

        for json_file in json_files:
            if json_file.stem.endswith("_raw"):
                json_path = json_file
                if base_filename is None:
                    base_filename = json_file.stem[:-4]

        if base_filename and (text_path or json_path):
            # Get file modification time
            if text_path and text_path.exists():
                mtime = datetime.fromtimestamp(text_path.stat().st_mtime)
            elif json_path and json_path.exists():
                mtime = datetime.fromtimestamp(json_path.stat().st_mtime)
            else:
                mtime = datetime.utcnow()

            job_data = {
                "job_id": job_id,
                "filename": base_filename,
                "status": "completed",
                "progress": 100,
                "message": "Previously processed",
                "created_at": mtime.isoformat(),
                "updated_at": mtime.isoformat(),
            }

            if text_path and text_path.exists():
                job_data["text_path"] = str(text_path.absolute())
                job_data["text_url"] = f"/download/{job_id}/text"

            if json_path and json_path.exists():
                job_data["json_path"] = str(json_path.absolute())
                job_data["json_url"] = f"/download/{job_id}/json"

            if summary_path and summary_path.exists():
                job_data["summary_path"] = str(summary_path.absolute())
                job_data["summary_url"] = f"/download/{job_id}/summary"

            if m4a_path and m4a_path.exists():
                job_data["m4a_path"] = str(m4a_path.absolute())
                job_data["m4a_url"] = f"/download/{job_id}/m4a"

            if flac_path and flac_path.exists():
                job_data["flac_path"] = str(flac_path.absolute())
                job_data["flac_url"] = f"/download/{job_id}/flac"

            with jobs_lock:
                jobs[job_id] = job_data

            loaded_count += 1

    if loaded_count > 0:
        print(f"Loaded {loaded_count} existing job(s) from output directory", flush=True)


@socketio.on("connect")
def handle_connect():
    """Handle WebSocket connection."""
    emit("connected", {"message": "Connected to server"})


def main():
    """Run the Flask web app."""
    ensure_dir(app.config["UPLOAD_FOLDER"])
    ensure_dir(app.config["OUTPUT_FOLDER"])
    ensure_dir(app.config["MODEL_CACHE"])

    print("Loading existing jobs...", flush=True)
    load_existing_jobs()

    print("Initializing transcription model...", flush=True)
    init_model()

    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"Starting web server on {host}:{port}", flush=True)
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
