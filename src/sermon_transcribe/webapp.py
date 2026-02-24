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

from flask import Flask, jsonify, render_template, request, send_file, redirect, url_for, session
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from urllib.parse import urlparse, urljoin

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
app.config["SESSION_COOKIE_SECURE"] = False  # Set to True in production with HTTPS
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = 3600  # 1 hour session timeout

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."

# User credentials from environment variables
users = {}
auth_username = os.environ.get("AUTH_USERNAME")
auth_password = os.environ.get("AUTH_PASSWORD")

if auth_username and auth_password:
    # Store hashed password for security
    users[auth_username] = generate_password_hash(auth_password)
    print(f"Authentication enabled for user: {auth_username}", flush=True)
else:
    print("Warning: No authentication configured (AUTH_USERNAME and AUTH_PASSWORD not set)", flush=True)
    print("Application is running in OPEN ACCESS mode - not suitable for public deployment!", flush=True)


# Simple User class for Flask-Login
class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    """Load user from session."""
    if user_id in users or not users:
        return User(user_id)
    return None


@login_manager.unauthorized_handler
def unauthorized():
    """Redirect unauthorized users to login page."""
    return redirect(url_for("login"))


def is_safe_url(target):
    """Check if a redirect target URL is safe (prevents open redirects)."""
    if not target:
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=4 * 1024 * 1024 * 1024,  # 4GB max for uploads
    async_mode='threading'  # Use threading instead of eventlet
)

# Global state for jobs
jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()

# Cancelled jobs tracking
cancelled_jobs = set()
cancelled_lock = threading.Lock()

# Processing semaphore - ensures only one transcription runs at a time
processing_semaphore = threading.Semaphore(1)

# Login attempt tracking for rate limiting (simple in-memory tracking)
login_attempts = {}
login_attempts_lock = threading.Lock()

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

    # Get default language from environment (defaults to "en" for English)
    default_language = os.environ.get("WHISPER_LANGUAGE", "en")
    # Convert empty string or "auto" to None for auto-detection
    if default_language in ("", "auto", "none", "None"):
        default_language = None

    config = build_config(
        model=model_name,
        device=device,
        compute_type=compute_type,
        beam_size=5,
        language=default_language,
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
            m4a_path = output_dir / f"{base_name}.m4a"

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
                except json.JSONDecodeError:
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
            flac_path = output_dir / f"{base_name}.flac"

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
                except json.JSONDecodeError:
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


def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    with cancelled_lock:
        return job_id in cancelled_jobs


def process_transcription(job_id: str, audio_path: Path, output_dir: Path, do_cleanup: bool, convert_m4a: bool = False, convert_flac: bool = False, language: Optional[str] = None):
    """Background task for transcribing audio."""
    # Small delay to ensure client receives upload response before WebSocket updates
    time.sleep(0.2)

    # Wait for semaphore (ensures sequential processing)
    emit_progress(job_id, "queued", 5, "Waiting in processing queue...")

    with processing_semaphore:
        try:
            # Check for cancellation before starting
            if is_job_cancelled(job_id):
                emit_progress(job_id, "cancelled", 0, "Job cancelled by user")
                return

            emit_progress(job_id, "processing", 10, "Starting transcription...")
            emit_progress(job_id, "processing", 15, "Transcribing audio... (this may take several minutes)")

            # Create a config with the specified language (or use default)
            transcription_config = config
            if language is not None and language != config.language:
                from sermon_transcribe.transcription import build_config
                transcription_config = build_config(
                    model=config.model,
                    device=config.device,
                    compute_type=config.compute_type,
                    beam_size=config.beam_size,
                    language=language,
                    task=config.task,
                    vad_filter=config.vad_filter,
                    cache_dir=config.cache_dir,
                    hf_token=config.hf_token,
                )

            result = transcribe_file(
                model=model,
                source_path=audio_path,
                output_dir=output_dir,
                config=transcription_config,
                convert_flac=False,
                raw_suffix="_timestamps" if do_cleanup else "",
            )

            # Check for cancellation after transcription
            if is_job_cancelled(job_id):
                emit_progress(job_id, "cancelled", 0, "Job cancelled by user")
                return

            emit_progress(job_id, "processing", 50, "Transcription complete")

            # Set initial paths (cleanup_transcript will update text_path if it runs)
            with jobs_lock:
                jobs[job_id]["text_path"] = str(result.text_path.absolute())
                jobs[job_id]["json_path"] = str(result.json_path.absolute())

            # Perform audio conversion if requested
            conversion_results = {}
            if convert_m4a or convert_flac:
                # Check for cancellation before conversion
                if is_job_cancelled(job_id):
                    emit_progress(job_id, "cancelled", 0, "Job cancelled by user")
                    return
                conversion_results = convert_audio(job_id, audio_path, output_dir, convert_m4a, convert_flac)

            # Add conversion results to job
            with jobs_lock:
                jobs[job_id].update(conversion_results)

            if do_cleanup:
                # Check for cancellation before cleanup
                if is_job_cancelled(job_id):
                    emit_progress(job_id, "cancelled", 0, "Job cancelled by user")
                    return
                emit_progress(job_id, "processing", 60, "Cleaning transcript with Claude...")
                cleanup_transcript(job_id, result, output_dir)

            # Check for cancellation before completing
            if is_job_cancelled(job_id):
                emit_progress(job_id, "cancelled", 0, "Job cancelled by user")
                return

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
            # Don't report error if job was cancelled
            if is_job_cancelled(job_id):
                emit_progress(job_id, "cancelled", 0, "Job cancelled by user")
            else:
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
        # Check for cancellation before processing each chunk
        if is_job_cancelled(job_id):
            return

        emit_progress(job_id, "processing", 65 + (idx * 10 // len(chunks)), f"Cleaning chunk {idx}/{len(chunks)}")
        prompt = build_prompt(chunk)
        try:
            cleaned = call_claude(
                prompt=prompt,
                api_key=api_key,
                model=model_name,
                max_tokens=4000,  # Increased from 1200 to prevent truncation
                temperature=0.1,
                timeout=120,
            )
            cleaned_text_chunk = cleaned.strip()

            # Warn if chunk came back empty but original wasn't
            if not cleaned_text_chunk and chunk.strip():
                print(f"WARNING: Chunk {idx}/{len(chunks)} returned empty from Claude (original had {len(chunk)} chars)", flush=True)
                # Keep original chunk if Claude returns nothing
                cleaned_chunks.append(chunk.strip())
            else:
                cleaned_chunks.append(cleaned_text_chunk)
        except Exception as e:
            # Check if error is due to cancellation
            if is_job_cancelled(job_id):
                return
            print(f"ERROR: Chunk {idx}/{len(chunks)} failed to clean: {str(e)}", flush=True)
            # Fall back to original chunk on error
            cleaned_chunks.append(chunk.strip())

    # Join all chunks, but verify we haven't lost content
    cleaned_text = "\n\n".join(chunk for chunk in cleaned_chunks if chunk)

    # Log if we lost chunks
    non_empty_chunks = sum(1 for c in cleaned_chunks if c)
    if non_empty_chunks < len(chunks):
        print(f"WARNING: Lost {len(chunks) - non_empty_chunks} chunks during cleaning", flush=True)
    base_name = result.audio_path.stem
    cleaned_path = output_dir / f"{base_name}.txt"
    cleaned_path.write_text(apply_disclaimer(cleaned_text), encoding="utf-8")

    # Generate summary
    emit_progress(job_id, "processing", 80, "Generating summary...")

    # Check for cancellation before summary
    if is_job_cancelled(job_id):
        return

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
            # Check for cancellation before processing each summary chunk
            if is_job_cancelled(job_id):
                return

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


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login with rate limiting."""
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        # Get client IP for rate limiting
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        if client_ip:
            client_ip = client_ip.split(",")[0].strip()

        # Rate limiting: max 5 failed attempts per IP within 15 minutes
        with login_attempts_lock:
            now = time.time()
            if client_ip in login_attempts:
                attempts = login_attempts[client_ip]
                # Clean old attempts (older than 15 minutes)
                attempts = [t for t in attempts if now - t < 900]

                if len(attempts) >= 5:
                    return render_template("login.html",
                        error="Too many failed login attempts. Please try again in 15 minutes."), 429

                login_attempts[client_ip] = attempts
            else:
                login_attempts[client_ip] = []

        # If no users configured, allow any login (backward compatibility)
        if not users:
            user = User(username if username else "anonymous")
            login_user(user)
            # Clear rate limit on successful login
            with login_attempts_lock:
                if client_ip in login_attempts:
                    login_attempts[client_ip] = []
            next_page = request.args.get("next")
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            return redirect(url_for("index"))

        # Validate credentials
        if username in users and check_password_hash(users[username], password):
            user = User(username)
            login_user(user)
            # Clear rate limit on successful login
            with login_attempts_lock:
                if client_ip in login_attempts:
                    login_attempts[client_ip] = []
            next_page = request.args.get("next")
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            return redirect(url_for("index"))
        else:
            # Record failed attempt
            with login_attempts_lock:
                if client_ip in login_attempts:
                    login_attempts[client_ip].append(now)
                else:
                    login_attempts[client_ip] = [now]

            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    """Main upload page."""
    response = app.make_response(render_template("index.html"))
    # Prevent browser caching to ensure users always get latest version
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/review/<job_id>")
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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


@app.route("/api/audio/<job_id>/trim", methods=["POST"])
@login_required
def trim_audio(job_id: str):
    """Trim audio file and optionally re-transcribe."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Get trim parameters
    data = request.get_json()
    start_time = float(data.get("start_time", 0))
    end_time = data.get("end_time")

    if end_time is None:
        return jsonify({"error": "end_time is required"}), 400

    end_time = float(end_time)

    if start_time < 0 or end_time <= start_time:
        return jsonify({"error": "Invalid trim times"}), 400

    # Find audio file
    upload_dir = app.config["UPLOAD_FOLDER"] / job_id
    if not upload_dir.exists():
        return jsonify({"error": "Audio not found"}), 404

    wav_files = list(upload_dir.glob("*.WAV")) + list(upload_dir.glob("*.wav"))
    if not wav_files:
        return jsonify({"error": "Audio file not found"}), 404

    audio_path = wav_files[0]

    # Create backup of original audio
    backup_path = audio_path.with_suffix(audio_path.suffix + ".backup")
    if not backup_path.exists():
        shutil.copy2(audio_path, backup_path)

    # Create temporary output path
    temp_output = audio_path.with_suffix(".trimmed.wav")

    # Calculate duration
    duration = end_time - start_time

    # Use ffmpeg to trim the audio
    try:
        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(duration),
            "-c", "copy",
            "-y",
            str(temp_output)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Replace original with trimmed version
        shutil.move(str(temp_output), str(audio_path))

        # Update segments timestamps by subtracting start_time
        output_dir = app.config["OUTPUT_FOLDER"] / job_id
        if output_dir.exists():
            json_files = list(output_dir.glob("*_timestamps.json"))
            if json_files:
                json_path = json_files[0]
                try:
                    with open(json_path, 'r') as f:
                        segments_data = json.load(f)

                    # Update timestamps in segments
                    if "segments" in segments_data:
                        for segment in segments_data["segments"]:
                            segment["start"] = max(0, segment["start"] - start_time)
                            segment["end"] = max(0, segment["end"] - start_time)

                        # Filter out segments that are now outside the trimmed range
                        segments_data["segments"] = [
                            s for s in segments_data["segments"]
                            if s["start"] < duration
                        ]

                        # Save updated segments
                        with open(json_path, 'w') as f:
                            json.dump(segments_data, f, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to update segment timestamps: {e}", flush=True)

        return jsonify({
            "success": True,
            "message": f"Audio trimmed successfully. New duration: {duration:.2f}s",
            "new_duration": duration,
            "backup_created": True
        })

    except subprocess.CalledProcessError as e:
        # Clean up temp file if it exists
        if temp_output.exists():
            temp_output.unlink()

        return jsonify({
            "error": f"Failed to trim audio: {e.stderr}"
        }), 500
    except Exception as e:
        # Clean up temp file if it exists
        if temp_output.exists():
            temp_output.unlink()

        return jsonify({
            "error": f"Failed to trim audio: {str(e)}"
        }), 500


@app.route("/api/segments/<job_id>")
@login_required
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

    # Find raw JSON file with segments (try both new and old naming)
    json_files = list(output_dir.glob("*_timestamps.json"))
    if not json_files:
        json_files = list(output_dir.glob("*_raw.json"))  # Fallback to old naming
    if not json_files:
        return jsonify({"error": "Segments not found"}), 404

    json_path = json_files[0]

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to load segments: {str(e)}"}), 500


# Track chunked uploads
chunked_uploads: Dict[str, dict] = {}
chunked_uploads_lock = threading.Lock()

# Cleanup configuration
CHUNKED_UPLOAD_TIMEOUT = 30 * 60  # 30 minutes in seconds


def cleanup_stale_uploads():
    """Background task to clean up abandoned chunked uploads."""
    while True:
        try:
            time.sleep(5 * 60)  # Run cleanup every 5 minutes

            current_time = datetime.utcnow()
            stale_uploads = []

            with chunked_uploads_lock:
                for upload_id, upload_info in chunked_uploads.items():
                    age_seconds = (current_time - upload_info["created_at"]).total_seconds()

                    # Mark as stale if:
                    # 1. Older than timeout AND not complete
                    # 2. Not currently being assembled
                    if age_seconds > CHUNKED_UPLOAD_TIMEOUT:
                        chunks_received = len(upload_info["received_chunks"])
                        total_chunks = upload_info["total_chunks"]

                        if chunks_received < total_chunks and not upload_info.get("assembling"):
                            stale_uploads.append(upload_id)

            # Clean up stale uploads outside the lock
            for upload_id in stale_uploads:
                with chunked_uploads_lock:
                    upload_info = chunked_uploads.get(upload_id)
                    if not upload_info:
                        continue

                    print(f"Cleaning up stale upload: {upload_id} (age: {age_seconds/60:.1f} min, chunks: {len(upload_info['received_chunks'])}/{upload_info['total_chunks']})", flush=True)

                    # Delete partial chunk files
                    try:
                        for i in range(upload_info["total_chunks"]):
                            chunk_path = upload_info["audio_path"].with_suffix(f".part{i}")
                            if chunk_path.exists():
                                chunk_path.unlink()
                    except Exception as e:
                        print(f"Error cleaning up chunks for {upload_id}: {e}", flush=True)

                    # Remove from tracking
                    del chunked_uploads[upload_id]

        except Exception as e:
            print(f"Error in cleanup_stale_uploads: {e}", flush=True)


@app.route("/upload/chunk", methods=["POST"])
@login_required
def upload_chunk():
    """Handle chunked file upload."""
    if "chunk" not in request.files:
        return jsonify({"error": "No chunk provided"}), 400

    chunk = request.files["chunk"]
    upload_id = request.form.get("upload_id")

    # Validate upload_id format (should be alphanumeric with underscores)
    if not upload_id or not upload_id.replace("_", "").isalnum():
        return jsonify({"error": "Invalid upload_id"}), 400

    # Parse and validate chunk parameters
    try:
        chunk_index = int(request.form.get("chunk_index", 0))
        total_chunks = int(request.form.get("total_chunks", 1))
    except ValueError:
        return jsonify({"error": "Invalid chunk parameters"}), 400

    # Validate chunk index and total chunks
    if chunk_index < 0 or total_chunks <= 0:
        return jsonify({"error": "Invalid chunk index or total chunks"}), 400

    if chunk_index >= total_chunks:
        return jsonify({"error": "Chunk index exceeds total chunks"}), 400

    # Prevent DoS with excessive chunks (limit to 10,000 chunks = ~20GB file at 2MB/chunk)
    if total_chunks > 10000:
        return jsonify({"error": "File too large (too many chunks)"}), 400

    # Sanitize filename to prevent path traversal attacks
    raw_filename = request.form.get("filename", "audio.wav")
    filename = secure_filename(raw_filename)

    if not filename:
        # If secure_filename returns empty (e.g., filename was "../../../etc/passwd"), use default
        filename = "audio.wav"

    # Initialize chunked upload tracking
    with chunked_uploads_lock:
        if upload_id not in chunked_uploads:
            upload_path = app.config["UPLOAD_FOLDER"] / upload_id
            ensure_dir(upload_path)
            audio_path = upload_path / filename

            chunked_uploads[upload_id] = {
                "filename": filename,
                "audio_path": audio_path,
                "total_chunks": total_chunks,
                "received_chunks": set(),
                "created_at": datetime.utcnow(),
                "assembling": False,  # Prevent race condition
            }

        upload_info = chunked_uploads[upload_id]

        # Validate chunk isn't already received (prevent duplicate uploads)
        if chunk_index in upload_info["received_chunks"]:
            chunks_received = len(upload_info["received_chunks"])
            return jsonify({
                "status": "duplicate",
                "upload_id": upload_id,
                "chunks_received": chunks_received,
                "total_chunks": total_chunks,
                "message": f"Chunk {chunk_index + 1} already received"
            })

        # Save chunk to temporary file
        chunk_path = upload_info["audio_path"].with_suffix(f".part{chunk_index}")
        try:
            chunk.save(str(chunk_path))
            upload_info["received_chunks"].add(chunk_index)
        except Exception as e:
            return jsonify({"error": f"Failed to save chunk: {str(e)}"}), 500

        chunks_received = len(upload_info["received_chunks"])

        # Check if all chunks received
        if chunks_received == total_chunks:
            # Prevent duplicate assembly attempts (race condition)
            if upload_info["assembling"]:
                return jsonify({
                    "status": "assembling",
                    "upload_id": upload_id,
                    "message": "File is being assembled"
                }), 202  # 202 Accepted

            upload_info["assembling"] = True

            # Copy path info before releasing lock for assembly
            audio_path = upload_info["audio_path"]
            assembly_total_chunks = total_chunks
        else:
            # Return status for incomplete upload
            return jsonify({
                "status": "receiving",
                "upload_id": upload_id,
                "chunks_received": chunks_received,
                "total_chunks": total_chunks,
                "message": f"Chunk {chunk_index + 1}/{total_chunks} received"
            })

    # Perform assembly outside the lock (slow I/O operation)
    # Only reaches here if chunks_received == total_chunks
    try:
        with open(audio_path, "wb") as output_file:
            for i in range(assembly_total_chunks):
                chunk_path = audio_path.with_suffix(f".part{i}")
                if not chunk_path.exists():
                    raise FileNotFoundError(f"Chunk {i} missing during assembly")

                # Stream copy to avoid loading entire chunk into memory
                with open(chunk_path, "rb") as chunk_file:
                    shutil.copyfileobj(chunk_file, output_file, length=1024*1024)  # 1MB buffer

                # Delete chunk file after appending
                chunk_path.unlink()

        return jsonify({
            "status": "complete",
            "upload_id": upload_id,
            "chunks_received": chunks_received,
            "total_chunks": assembly_total_chunks,
            "message": "All chunks received, file assembled"
        })

    except Exception as e:
        # Clean up partial files
        try:
            if audio_path.exists():
                audio_path.unlink()
        except:
            pass

        for i in range(assembly_total_chunks):
            chunk_path = audio_path.with_suffix(f".part{i}")
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
            except:
                pass

        # Mark as not assembling so client can retry
        with chunked_uploads_lock:
            if upload_id in chunked_uploads:
                chunked_uploads[upload_id]["assembling"] = False

        return jsonify({"error": f"Failed to assemble file: {str(e)}"}), 500


@app.route("/upload/start", methods=["POST"])
@login_required
def start_transcription():
    """Start transcription after chunked upload completes."""
    data = request.get_json()
    upload_id = data.get("upload_id")

    if not upload_id:
        return jsonify({"error": "No upload_id provided"}), 400

    with chunked_uploads_lock:
        upload_info = chunked_uploads.get(upload_id)

    if not upload_info:
        return jsonify({"error": "Upload not found"}), 404

    if len(upload_info["received_chunks"]) != upload_info["total_chunks"]:
        return jsonify({"error": "Upload incomplete"}), 400

    # Get transcription options
    do_cleanup = data.get("cleanup", False)
    convert_m4a = data.get("convert_m4a", False)
    convert_flac = data.get("convert_flac", False)
    language = data.get("language", "")

    if language in ("", "auto", "none"):
        language = None

    # Use upload_id as job_id
    job_id = upload_id
    audio_path = upload_info["audio_path"]
    filename = upload_info["filename"]

    # Create output directory
    output_dir = app.config["OUTPUT_FOLDER"] / job_id
    ensure_dir(output_dir)

    # Initialize job
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "message": "Job queued",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "cleanup": do_cleanup,
            "language": language or config.language,
        }

    # Start background processing
    thread = threading.Thread(
        target=process_transcription,
        args=(job_id, audio_path, output_dir, do_cleanup, convert_m4a, convert_flac, language),
        daemon=True,
    )
    thread.start()

    # Clean up chunked upload tracking
    with chunked_uploads_lock:
        del chunked_uploads[upload_id]

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/upload", methods=["POST"])
@login_required
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

    # Get language parameter (defaults to environment setting, which defaults to "en")
    language = request.form.get("language", "")
    if language in ("", "auto", "none"):
        language = None  # Auto-detect

    # Generate job ID
    job_id = secrets.token_hex(16)

    # Save uploaded file
    ensure_dir(app.config["UPLOAD_FOLDER"])
    upload_path = app.config["UPLOAD_FOLDER"] / job_id
    ensure_dir(upload_path)

    # Sanitize filename to prevent path traversal attacks
    audio_filename = secure_filename(file.filename) if file.filename else "audio.wav"
    if not audio_filename:
        audio_filename = "audio.wav"
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
            "language": language or config.language,  # Store language used
        }

    # Start background processing
    thread = threading.Thread(
        target=process_transcription,
        args=(job_id, audio_path, output_dir, do_cleanup, convert_m4a, convert_flac, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/retranscribe/<job_id>", methods=["POST"])
@login_required
def retranscribe(job_id: str):
    """Re-transcribe an existing job with optional new language setting."""
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404

        old_job = jobs[job_id].copy()

    # Get options from request
    data = request.get_json() or {}
    do_cleanup = data.get("cleanup", old_job.get("cleanup", False))
    language = data.get("language", old_job.get("language"))

    # Convert empty string or "auto" to None
    if language in ("", "auto", "none"):
        language = None

    # Find the original audio file
    upload_path = app.config["UPLOAD_FOLDER"] / job_id
    if not upload_path.exists():
        return jsonify({"error": "Original audio file not found"}), 404

    # Find audio file in upload directory
    audio_files = list(upload_path.glob("*"))
    if not audio_files:
        return jsonify({"error": "Original audio file not found"}), 404

    audio_path = audio_files[0]  # Use first file found

    # Use existing output directory (preserves M4A/FLAC conversions)
    output_dir = app.config["OUTPUT_FOLDER"] / job_id

    # Preserve audio conversion URLs from old job
    preserved_fields = {}
    for key in ['m4a_url', 'flac_url', 'm4a_path', 'flac_path']:
        if key in old_job:
            preserved_fields[key] = old_job[key]

    # Update existing job instead of creating new one
    with jobs_lock:
        jobs[job_id].update({
            "status": "queued",
            "progress": 0,
            "message": "Re-transcription queued",
            "updated_at": datetime.utcnow().isoformat(),
            "cleanup": do_cleanup,
            "language": language or config.language,
        })
        # Preserve audio conversions
        jobs[job_id].update(preserved_fields)

    # Start background processing (don't convert audio again - set to False)
    thread = threading.Thread(
        target=process_transcription,
        args=(job_id, audio_path, output_dir, do_cleanup, False, False, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued", "message": "Re-transcription started"})


@app.route("/upload-transcript", methods=["POST"])
@login_required
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

    # Save raw transcript - sanitize filename to prevent path traversal
    safe_filename = secure_filename(file.filename) if file.filename else "transcript.txt"
    if not safe_filename:
        safe_filename = "transcript.txt"
    transcript_filename = Path(safe_filename).stem
    raw_suffix = "_timestamps" if do_cleanup else ""
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
@login_required
def status(job_id: str):
    """Get job status."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job)


@app.route("/jobs")
@login_required
def list_jobs():
    """List all jobs."""
    with jobs_lock:
        job_list = list(jobs.values())

    # Sort by creation time, newest first
    job_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return jsonify({"jobs": job_list})


@app.route("/download/<job_id>/<file_type>")
@login_required
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
    elif file_type == "all":
        # Create a zip file with all job files
        import zipfile
        import tempfile

        with jobs_lock:
            job = jobs.get(job_id)

        if not job:
            return jsonify({"error": "Job not found"}), 404

        # Create temporary zip file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')

        try:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add text file
                if "text_path" in job and Path(job["text_path"]).exists():
                    zipf.write(job["text_path"], Path(job["text_path"]).name)

                # Add JSON file
                if "json_path" in job and Path(job["json_path"]).exists():
                    zipf.write(job["json_path"], Path(job["json_path"]).name)

                # Add summary file
                if "summary_path" in job and Path(job["summary_path"]).exists():
                    zipf.write(job["summary_path"], Path(job["summary_path"]).name)

                # Add M4A file
                if "m4a_path" in job and Path(job["m4a_path"]).exists():
                    zipf.write(job["m4a_path"], Path(job["m4a_path"]).name)

                # Add FLAC file
                if "flac_path" in job and Path(job["flac_path"]).exists():
                    zipf.write(job["flac_path"], Path(job["flac_path"]).name)

            # Get base filename for the zip
            base_name = job.get("filename", "transcription")
            if base_name.endswith(('.mp3', '.wav', '.m4a', '.flac')):
                base_name = Path(base_name).stem

            zip_filename = f"{base_name}_complete.zip"

            return send_file(
                temp_zip.name,
                as_attachment=True,
                download_name=zip_filename,
                mimetype='application/zip'
            )
        except Exception as e:
            if Path(temp_zip.name).exists():
                Path(temp_zip.name).unlink()
            return jsonify({"error": f"Failed to create zip: {str(e)}"}), 500
    else:
        return jsonify({"error": "File not found"}), 404


@app.route("/api/log-error", methods=["POST"])
@login_required
def log_error():
    """Log client-side JavaScript errors for debugging."""
    data = request.get_json()
    print(f"CLIENT ERROR from {request.remote_addr}:", flush=True)
    print(f"  Message: {data.get('message')}", flush=True)
    print(f"  User-Agent: {data.get('userAgent')}", flush=True)
    print(f"  URL: {data.get('url')}", flush=True)
    if data.get('stack'):
        print(f"  Stack: {data.get('stack')}", flush=True)
    return jsonify({"success": True})


@app.route("/job/<job_id>/cancel", methods=["POST"])
@login_required
def cancel_job(job_id: str):
    """Cancel a running or queued job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        # Only allow cancellation of queued or processing jobs
        if job["status"] not in ["queued", "processing"]:
            return jsonify({"error": f"Cannot cancel job with status: {job['status']}"}), 400

    # Add job to cancelled set
    with cancelled_lock:
        cancelled_jobs.add(job_id)

    # Update job status
    emit_progress(job_id, "cancelling", job.get("progress", 0), "Cancelling job...")

    return jsonify({"success": True, "message": "Job cancellation requested"})


@app.route("/job/<job_id>", methods=["DELETE"])
@login_required
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

        # Remove from cancelled set if present
        with cancelled_lock:
            cancelled_jobs.discard(job_id)

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
        m4a_files = list(job_dir.glob("*.m4a"))
        flac_files = list(job_dir.glob("*.flac"))

        if m4a_files:
            m4a_path = m4a_files[0]
        if flac_files:
            flac_path = flac_files[0]

        # Look for files
        for txt_file in txt_files:
            if txt_file.stem.endswith("_summary"):
                summary_path = txt_file
            elif txt_file.stem.endswith("_timestamps"):
                # Check if cleaned version exists
                cleaned_name = txt_file.stem[:-11] + ".txt"  # Remove _timestamps
                cleaned_path = txt_file.parent / cleaned_name
            elif txt_file.stem.endswith("_raw"):
                # Legacy naming - check if cleaned version exists
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
            if json_file.stem.endswith("_timestamps"):
                json_path = json_file
                if base_filename is None:
                    base_filename = json_file.stem[:-11]
            elif json_file.stem.endswith("_raw"):
                # Legacy naming
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

    # Start background cleanup thread for stale chunked uploads
    print("Starting chunked upload cleanup thread...", flush=True)
    cleanup_thread = threading.Thread(target=cleanup_stale_uploads, daemon=True)
    cleanup_thread.start()

    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"Starting web server on {host}:{port}", flush=True)
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
