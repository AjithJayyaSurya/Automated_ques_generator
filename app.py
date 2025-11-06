import streamlit as st
import pandas as pd
import random
import ast
import re
import spacy
from collections import Counter
import tempfile
import os
import PyPDF2
from docx import Document
from fpdf import FPDF
import base64

# Initialize session state
if 'mcq_df' not in st.session_state:
    st.session_state.mcq_df = pd.DataFrame(columns=['Question', 'Options', 'Answer'])
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'student_answers' not in st.session_state:
    st.session_state.student_answers = {}
if 'student_score' not in st.session_state:
    st.session_state.student_score = 0
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
        return None

nlp = load_spacy_model()

# Text Analytics Engine
class TextAnalyticsEngine:
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'it', 'its', 'they', 'them',
            'their', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our'
        }

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def extract_keywords(self, text, top_n=10):
        if nlp:
            doc = nlp(text)
            keywords = []
            
            # Extract named entities first
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "NORP"]:
                    clean_ent = re.sub(r'[^\w\s]', '', ent.text).lower()
                    if clean_ent not in self.stop_words and len(clean_ent) > 2:
                        keywords.append(clean_ent)
                        if len(keywords) >= top_n:
                            return list(set(keywords))

            # Extract important nouns and adjectives
            word_freq = Counter()
            for token in doc:
                if (not token.is_stop and not token.is_punct and 
                    token.is_alpha and len(token.text) > 2):
                    if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"]:
                        clean_word = token.lemma_.lower()
                        if clean_word not in self.stop_words:
                            word_freq[clean_word] += 1

            # Add most common words
            for word, freq in word_freq.most_common(top_n):
                if word not in keywords:
                    keywords.append(word)
                    if len(keywords) >= top_n:
                        break
            
            return list(set(keywords[:top_n]))
        else:
            words = self.preprocess_text(text).split()
            filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
            word_freq = Counter(filtered_words)
            keywords = [word for word, freq in word_freq.most_common(top_n)]
            return keywords

    def sentence_tokenize(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def generate_mcq_questions(self, text, num_questions=20):
        sentences = self.sentence_tokenize(text)
        questions = []
        used_sentences = set()

        for sentence in sentences:
            if len(questions) >= num_questions:
                break
                
            if len(sentence) > 25 and sentence not in used_sentences:
                keywords = self.extract_keywords(sentence, top_n=5)
                
                if len(keywords) >= 2:
                    # Try different keywords for blank
                    for keyword in keywords:
                        if len(questions) >= num_questions:
                            break
                            
                        if len(keyword) > 3:  # Ensure keyword is substantial
                            # Create question with blank
                            question_text = re.sub(
                                r'\b' + re.escape(keyword) + r'\b', 
                                "______", 
                                sentence, 
                                1, 
                                re.IGNORECASE
                            )
                            
                            # Only use if blank was actually created
                            if "______" in question_text:
                                correct_answer = keyword
                                distractors = self.generate_distractors(keywords, correct_answer)
                                
                                options = [correct_answer] + distractors
                                random.shuffle(options)
                                
                                # Ensure we have exactly 4 options
                                while len(options) < 4:
                                    generic_option = f"option_{len(options)+1}"
                                    options.append(generic_option)
                                options = options[:4]
                                
                                questions.append({
                                    'question': question_text,
                                    'options': options,
                                    'answer': correct_answer
                                })
                                used_sentences.add(sentence)
                                break

        # If we don't have enough questions, create simpler ones
        if len(questions) < num_questions:
            additional_needed = num_questions - len(questions)
            simple_sentences = [s for s in sentences if s not in used_sentences and len(s) > 15]
            
            for sentence in simple_sentences[:additional_needed]:
                words = sentence.split()
                important_words = [w for w in words if len(w) > 4 and w.lower() not in self.stop_words]
                
                if important_words:
                    keyword = random.choice(important_words)
                    question_text = sentence.replace(keyword, "______")
                    
                    # Create simple distractors
                    distractors = []
                    for w in important_words:
                        if w != keyword and len(distractors) < 3:
                            distractors.append(w)
                    
                    while len(distractors) < 3:
                        distractors.append(f"choice_{len(distractors)+1}")
                    
                    options = [keyword] + distractors
                    random.shuffle(options)
                    
                    questions.append({
                        'question': question_text,
                        'options': options[:4],
                        'answer': keyword
                    })

        return questions[:num_questions]

    def generate_distractors(self, keywords, correct_answer, num_distractors=3):
        distractors = []
        other_keywords = [kw for kw in keywords if kw.lower() != correct_answer.lower()]
        
        # Use other keywords first
        for kw in other_keywords:
            if len(distractors) < num_distractors:
                distractors.append(kw)
        
        # If still need more, create variations
        while len(distractors) < num_distractors:
            if other_keywords:
                base_word = random.choice(other_keywords)
                variations = [
                    base_word + 's',
                    base_word + 'ing',
                    'un' + base_word,
                    base_word + 'ed'
                ]
                for var in variations:
                    if len(distractors) < num_distractors and var not in distractors:
                        distractors.append(var)
                        break
        
        return distractors[:num_distractors]

# File reading functions
def read_txt_file(file):
    return file.read().decode("utf-8")

def read_pdf_file(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def read_docx_file(file):
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

# PDF Generation with UTF-8 support
def create_pdf(mcq_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add Unicode font (make sure to have DejaVuSans.ttf in your directory)
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
    except:
        pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "MCQ Question Paper", 0, 1, "C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    
    for i, row in mcq_df.iterrows():
        # Question
        question_text = f"Q{i+1}: {row['Question']}"
        # Handle Unicode characters by encoding properly
        try:
            pdf.multi_cell(0, 10, question_text.encode('latin-1', 'replace').decode('latin-1'))
        except:
            pdf.multi_cell(0, 10, f"Q{i+1}: [Question contains special characters]")
        
        pdf.ln(5)
        
        # Options
        try:
            options = ast.literal_eval(row['Options'])
            for j, option in enumerate(options):
                pdf.cell(20)  # Indent
                option_text = f"{chr(65+j)}. {option}"
                try:
                    pdf.multi_cell(0, 8, option_text.encode('latin-1', 'replace').decode('latin-1'))
                except:
                    pdf.multi_cell(0, 8, f"{chr(65+j)}. [Option contains special characters]")
        except Exception as e:
            pdf.multi_cell(0, 8, "A. Option A  B. Option B  C. Option C  D. Option D")
        
        pdf.ln(8)
    
    # Create download link
    pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(pdf_output.name)
    return pdf_output.name

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">📥 Download {file_label}</a>'
    return href

# Initialize engine
text_engine = TextAnalyticsEngine()

# Main App
def main():
    st.set_page_config(
        page_title="Automated Question Paper Generator",
        page_icon="🧠",
        layout="wide"
    )

    # Custom CSS - Removed white boxes
    st.markdown("""
    <style>
    .main-title {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .score-card {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .nav-button {
        width: 100%;
        margin: 5px 0;
    }
    /* Remove white boxes from options */
    .stRadio > div {
        background-color: transparent;
        border: none;
        padding: 0;
    }
    /* Style radio options to look clean */
    .stRadio [data-testid="stMarkdownContainer"] {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        transition: background-color 0.3s;
    }
    .stRadio [data-testid="stMarkdownContainer"]:hover {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title"><h1>🧠 Automated Question Paper Generator</h1><h3>Powered by Text Analytics & NLP Techniques</h3></div>', unsafe_allow_html=True)

    # Mode selection
    mode = st.sidebar.selectbox("Select Mode", ["👨‍🏫 Teacher Mode", "👨‍🎓 Student Mode"])

    if mode == "👨‍🏫 Teacher Mode":
        teacher_mode()
    else:
        student_mode()

def teacher_mode():
    st.header("👨‍🏫 Teacher Mode - Automated Question Generation")
    
    # File upload
    uploaded_file = st.file_uploader("Upload educational content", type=['txt', 'pdf', 'docx'])
    
    text_input = ""
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write("**File Details:**", file_details)
        
        # Read file content based on type
        if uploaded_file.type == "text/plain":
            text_input = read_txt_file(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            text_input = read_pdf_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text_input = read_docx_file(uploaded_file)
        
        # Show preview without white box
        with st.expander("📋 Preview Uploaded Content"):
            st.text_area("Content", text_input, height=200, label_visibility="collapsed")
    
    # Also allow direct text input
    direct_text = st.text_area(
        "**Or enter text directly:**",
        height=150,
        placeholder="Paste your educational text here (paragraphs, articles, textbook content)..."
    )
    
    if direct_text:
        text_input = direct_text
    
    if st.button("🚀 Generate 20 MCQ Questions", use_container_width=True, type="primary"):
        if text_input.strip():
            with st.spinner("🔍 Analyzing text and generating 20 high-quality questions..."):
                questions = text_engine.generate_mcq_questions(text_input, num_questions=20)
                if questions:
                    # Clear existing questions
                    st.session_state.mcq_df = pd.DataFrame(columns=['Question', 'Options', 'Answer'])
                    
                    # Display questions without white boxes
                    st.subheader("📝 Generated Questions")
                    for i, q in enumerate(questions, 1):
                        st.write(f"**Question {i}:** {q['question']}")
                        st.write("**Options:**")
                        for j, option in enumerate(q['options']):
                            st.write(f"   {chr(65+j)}. {option}")
                        st.write(f"**Answer:** {q['answer']}")
                        st.write("---")
                            
                        # Save to session state
                        new_row = {
                            'Question': q['question'],
                            'Options': str(q['options']),
                            'Answer': q['answer']
                        }
                        st.session_state.mcq_df = pd.concat([st.session_state.mcq_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    st.success(f"✅ Successfully generated {len(questions)} MCQ questions!")
                    
                    # Generate PDF
                    if len(st.session_state.mcq_df) > 0:
                        with st.spinner("📄 Creating PDF..."):
                            pdf_file = create_pdf(st.session_state.mcq_df)
                            st.markdown("---")
                            st.subheader("📥 Download Question Paper")
                            st.markdown(get_binary_file_downloader_html(pdf_file, 'MCQ_Question_Paper.pdf'), unsafe_allow_html=True)
                            os.unlink(pdf_file)  # Clean up temp file
                else:
                    st.error("❌ Could not generate questions from the provided text. Please try with different content.")
        else:
            st.error("❌ Please upload a file or enter some text first!")
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Database Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", len(st.session_state.mcq_df))
    with col2:
        status = "✅ Ready" if len(st.session_state.mcq_df) > 0 else "❌ Not Ready"
        st.metric("Student Mode", status)
    with col3:
        if len(st.session_state.mcq_df) > 0:
            st.metric("Last Generated", "Just now")
        else:
            st.metric("Last Generated", "Never")

def student_mode():
    st.header("👨‍🎓 Student Mode - Practice & Assessment")
    
    if len(st.session_state.mcq_df) == 0:
        st.warning("📝 No MCQ questions available. Please generate some questions in Teacher Mode first.")
        return
    
    # Reset if needed
    if st.button("🔄 Start New Attempt", type="secondary"):
        st.session_state.student_answers = {}
        st.session_state.student_score = 0
        st.session_state.current_index = 0
        st.session_state.submitted = False
        st.rerun()
    
    if not st.session_state.submitted:
        # Initialize student answers if not done
        if len(st.session_state.student_answers) != len(st.session_state.mcq_df):
            st.session_state.student_answers = {i: None for i in range(len(st.session_state.mcq_df))}
        
        # Progress bar
        progress = (st.session_state.current_index + 1) / len(st.session_state.mcq_df)
        st.progress(progress)
        st.write(f"**Progress: {st.session_state.current_index + 1} of {len(st.session_state.mcq_df)} questions**")
        
        # Navigation - Removed First and Last buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.current_index == 0):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()
        
        with col2:
            st.write(f"**Question {st.session_state.current_index + 1}**")
        
        with col3:
            if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.current_index == len(st.session_state.mcq_df) - 1):
                if st.session_state.current_index < len(st.session_state.mcq_df) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        # Question display - without white box
        current_q = st.session_state.mcq_df.iloc[st.session_state.current_index]
        
        st.subheader(f"Question {st.session_state.current_index + 1}")
        st.write(f"**{current_q['Question']}**")
        
        try:
            options = ast.literal_eval(current_q['Options'])
        except:
            options = ["Option A", "Option B", "Option C", "Option D"]
        
        # Get current answer
        current_answer = st.session_state.student_answers.get(st.session_state.current_index)
        
        # Display options using radio buttons without white boxes
        st.write("**Select your answer:**")
        
        # Use radio button for clean option selection
        selected_option = st.radio(
            "Choose your answer:",
            options,
            index=options.index(current_answer) if current_answer in options else 0,
            key=f"question_{st.session_state.current_index}",
            label_visibility="collapsed"
        )
        
        # Update answer if changed
        if selected_option != current_answer:
            st.session_state.student_answers[st.session_state.current_index] = selected_option
            st.rerun()
        
        # Show current selection
        if current_answer:
            st.info(f"✅ Your current selection: **{current_answer}**")
        
        # Submit button
        st.markdown("---")
        all_answered = all(answer is not None for answer in st.session_state.student_answers.values())
        
        if all_answered:
            if st.button("🎯 Submit All Answers", type="primary", use_container_width=True):
                # Calculate score
                score = 0
                for i in range(len(st.session_state.mcq_df)):
                    user_answer = st.session_state.student_answers[i]
                    correct_answer = st.session_state.mcq_df.iloc[i]['Answer']
                    if user_answer == correct_answer:
                        score += 1
                
                st.session_state.student_score = score
                st.session_state.submitted = True
                st.rerun()
        else:
            unanswered = sum(1 for answer in st.session_state.student_answers.values() if answer is None)
            st.warning(f"📋 You have {unanswered} unanswered questions. Please answer all questions before submitting.")
    
    else:
        # Results page
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.header("🎉 Assessment Complete!")
        st.subheader(f"Your Score: {st.session_state.student_score}/{len(st.session_state.mcq_df)}")
        percentage = (st.session_state.student_score / len(st.session_state.mcq_df)) * 100
        st.subheader(f"Percentage: {percentage:.1f}%")
        
        if percentage >= 80:
            st.success("🏆 Excellent Performance!")
        elif percentage >= 60:
            st.info("👍 Good Job!")
        elif percentage >= 40:
            st.warning("💪 Keep Practicing!")
        else:
            st.error("📚 Need More Practice!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed results
        st.subheader("📊 Detailed Results")
        
        for i in range(len(st.session_state.mcq_df)):
            user_answer = st.session_state.student_answers[i]
            correct_answer = st.session_state.mcq_df.iloc[i]['Answer']
            is_correct = user_answer == correct_answer
            
            with st.expander(f"Question {i+1} - {'✅ Correct' if is_correct else '❌ Incorrect'}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Question:** {st.session_state.mcq_df.iloc[i]['Question']}")
                    st.write(f"**Your answer:** {user_answer}")
                    st.write(f"**Correct answer:** {correct_answer}")
                with col2:
                    if is_correct:
                        st.success("✅ Correct")
                    else:
                        st.error("❌ Incorrect")

if __name__ == "__main__":
    main()