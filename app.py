from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from rag_engine import RAGEngine
from llm_client import GraniteClient

app = Flask(__name__)
app.secret_key = 'studymate_secret_key_2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Initialize components
rag_engine = RAGEngine(chunk_size=500, chunk_overlap=50)
llm_client = GraniteClient(device='cuda')  # GPU mode

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    session.clear()  # Clear previous session
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print(f"\n{'='*80}")
    print(f"[DEBUG] Starting PDF processing...")
    print(f"{'='*80}")
    
    if 'pdfs' not in request.files:
        print(f"[ERROR] No 'pdfs' in request.files")
        return redirect(url_for('index'))
    
    files = request.files.getlist('pdfs')
    print(f"[DEBUG] Received {len(files)} files")
    
    if not files or files[0].filename == '':
        print(f"[ERROR] Files list is empty or first file has no name")
        return redirect(url_for('index'))
    
    pdf_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pdf_paths.append(filepath)
            print(f"[DEBUG] Saved file: {filename} to {filepath}")
    
    if not pdf_paths:
        print(f"[ERROR] No valid PDF files found")
        return redirect(url_for('index'))
    
    # Process PDFs: extract, chunk, embed, build FAISS index
    try:
        print(f"[DEBUG] Processing {len(pdf_paths)} PDFs...")
        rag_engine.process_pdfs(pdf_paths)
        session['processed'] = True
        session['num_chunks'] = len(rag_engine.chunks)
        
        print(f"[DEBUG] RAG engine setup complete with {session['num_chunks']} chunks")
        
        # Clean up uploaded files
        for path in pdf_paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"[DEBUG] Cleaned up temporary file: {path}")
        
        print(f"[DEBUG] Processing complete!")
        print(f"{'='*80}\n")
        return render_template('ask.html', num_chunks=session['num_chunks'])
    except Exception as e:
        print(f"[ERROR] Exception during PDF processing: {str(e)}")
        import traceback
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
        print(f"\n{'='*80}")
        print(f"[DEBUG] Question received: {question}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve chunks
        print(f"[DEBUG] Total chunks available: {len(rag_engine.chunks)}")
        print(f"[DEBUG] Retrieving top-5 relevant chunks...")
        relevant_chunks = rag_engine.retrieve(question, top_k=5)
        
        print(f"[DEBUG] Retrieved {len(relevant_chunks)} chunks")
        for i, chunk in enumerate(relevant_chunks):
            print(f"[DEBUG] Chunk {i+1} relevance score: {chunk['score']:.4f}")
            print(f"[DEBUG] Chunk {i+1} text preview: {chunk['text'][:100]}...")
        
        if not relevant_chunks:
            print(f"[DEBUG] WARNING: No relevant chunks found!")
            answer_text = "Information not found in the document."
            sources = []
        else:
            # Step 2: Build context
            context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
            print(f"[DEBUG] Context length: {len(context)} characters")
            print(f"[DEBUG] Starting answer generation with Mistral model...")
            
            try:
                # Step 3: Generate answer
                answer_text = llm_client.generate_answer(question, context, max_new_tokens=150)
                print(f"[DEBUG] Answer generated successfully!")
                print(f"[DEBUG] Answer length: {len(answer_text)} characters")
                print(f"[DEBUG] Answer preview: {answer_text[:200]}...")
            except Exception as e:
                print(f"[ERROR] LLM generation failed: {str(e)}")
                import traceback
                print(traceback.format_exc())
                answer_text = f"Error in LLM generation: {str(e)}"
            
            # Step 4: Prepare sources
            sources = [
                {
                    'text': chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text'],
                    'score': f"{chunk['score']:.3f}"
                }
                for chunk in relevant_chunks
            ]
            print(f"[DEBUG] Sources prepared: {len(sources)} sources")
        
        print(f"[DEBUG] Rendering answer.html with answer text...")
        print(f"{'='*80}\n")
        return render_template('answer.html', 
                             question=question, 
                             answer=answer_text, 
                             sources=sources)
    
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Exception in /answer route:")
        import traceback
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        return f"Critical error: {str(e)}", 500

@app.route('/new_question')
def new_question():
    return redirect(url_for('ask_page'))

@app.route('/reset')
def reset():
    print(f"[DEBUG] Resetting RAG engine...")
    rag_engine.reset()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
