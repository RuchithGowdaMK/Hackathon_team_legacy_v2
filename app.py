from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from rag_engine import RAGEngine
from llm_client import GraniteClient

app = Flask(__name__)
app.secret_key = 'studymate_secret_key_2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Initialize components with improved settings
rag_engine = RAGEngine(chunk_size=250, chunk_overlap=120, debug=True)
llm_client = GraniteClient(device='cuda')  # GPU if available

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print("\n" + "="*90)
    print("[APP] Starting PDF processing...")
    print("="*90)

    if 'pdfs' not in request.files:
        print("[APP] ERROR: No 'pdfs' in request.files")
        return redirect(url_for('index'))

    files = request.files.getlist('pdfs')
    print(f"[APP] Received {len(files)} file(s)")

    if not files or files[0].filename == '':
        print("[APP] ERROR: Empty file list or first file has no name")
        return redirect(url_for('index'))

    pdf_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pdf_paths.append(filepath)
            print(f"[APP] Saved: {filepath}")

    if not pdf_paths:
        print("[APP] ERROR: No valid PDFs")
        return redirect(url_for('index'))

    try:
        print(f"[APP] Processing {len(pdf_paths)} PDF(s) with RAG...")
        rag_engine.process_pdfs(pdf_paths)
        session['processed'] = True
        session['num_chunks'] = len(rag_engine.chunks)
        print(f"[APP] RAG ready with {session['num_chunks']} chunks")

        # Clean up uploads (index is already built)
        for path in pdf_paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"[APP] Removed temp: {path}")

        print("[APP] Processing complete.\n")
        return render_template('ask.html', num_chunks=session['num_chunks'])
    except Exception as e:
        import traceback
        print("[APP] ERROR during processing:", e)
        print(traceback.format_exc())
        return f"Error processing PDFs: {str(e)}", 500

@app.route('/ask', methods=['GET'])
def ask_page():
    if not session.get('processed'):
        return redirect(url_for('index'))
    return render_template('ask.html', num_chunks=session.get('num_chunks', 0))

@app.route('/answer', methods=['POST'])
def answer():
    if not session.get('processed'):
        return redirect(url_for('index'))

    question = request.form.get('question', '').strip()
    if not question:
        return redirect(url_for('ask_page'))

    try:
        print("\n" + "="*90)
        print("[APP] Question:", question)
        print("="*90)

        # Retrieve + build tight context (includes re-ranking & boosting inside RAG)
        context, ranked = rag_engine.build_context(question, top_k=5, max_chars=3000)
        print(f"[APP] Context length: {len(context)}")
        print("[APP] Context preview >>>")
        print(context[:800], "...\n")

        if not context.strip():
            print("[APP] WARNING: Empty context.")
            answer_text = "Information not found in the document."
            sources = []
        else:
            # Generate with Granite
            print("[APP] Calling Granite to generate answer...")
            try:
                answer_text = llm_client.generate_answer(question, context, max_new_tokens=150)
                print("[APP] Answer OK. Preview:", answer_text[:200], "...\n")
            except Exception as e:
                import traceback
                print("[APP] ERROR in LLM generation:", e)
                print(traceback.format_exc())
                answer_text = f"Error in LLM generation: {str(e)}"

            # Prepare sources with metadata
            sources = []
            for r in ranked:
                meta = r.get('meta', {})
                src_name = os.path.basename(meta.get('source', ''))
                pages = f"{meta.get('page_start', '?')}-{meta.get('page_end', '?')}"
                sources.append({
                    'text': (r['text'][:300] + '...') if len(r['text']) > 300 else r['text'],
                    'score': f"{r.get('combined', r.get('score', 0.0)):.3f}",
                    'meta': f"{src_name} (pp. {pages})"
                })
            print(f"[APP] Prepared {len(sources)} source excerpt(s).")

        return render_template('answer.html', question=question, answer=answer_text, sources=sources)

    except Exception as e:
        import traceback
        print("[APP] CRITICAL:", e)
        print(traceback.format_exc())
        return f"Critical error: {str(e)}", 500

@app.route('/new_question')
def new_question():
    return redirect(url_for('ask_page'))

@app.route('/reset')
def reset():
    print("[APP] Resetting RAG engine...")
    rag_engine.reset()
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # In production set debug=False and use a proper WSGI server
    app.run(debug=True, host='0.0.0.0', port=5000)
