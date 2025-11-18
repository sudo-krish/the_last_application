import re
from pathlib import Path
from typing import List

import yaml
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # provider split
from langchain_community.vectorstores import FAISS         # integration split
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
# from langchain.retrievers.multi_query import MultiQueryRetriever   # optional multi-query
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
import dotenv

dotenv.load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "tla_config.yaml"
DATA_PATH = ROOT_DIR / "data" 
DATABASE_PATH = ROOT_DIR / "database" / "faiss_index"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def get_langchain_llm():
    if config["langchain"]["use_chatgpt"]:
        if 'gpt' in config["langchain"]["model_name"].lower():
            llm = ChatOpenAI(
                model=config["langchain"]["model_name"],
                temperature=config["langchain"]["temperature"],
            )
    elif config["langchain"]["using_ollama"]:
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=config["langchain"]["model_name"],
            temperature=config["langchain"]["temperature"],
        )
    return llm



class VectorRetriever:
    def __init__(self, model, data_path="data/", vectorstore_path="database/faiss_index"):
        self.data_path = data_path
        self.my_info_path = os.path.join(self.data_path, "my_info.txt")
        with open(self.my_info_path, "r") as f:
            self.my_info = f.read()
        self.vectorstore_path = vectorstore_path
        self.create_prompt()
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large") if config["langchain"]["using_ollama"] else OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = self.load_vectorstore()
        self.llm = model
        self.qa_chain = None

    def load_vectorstore(self):
        if not os.path.exists(self.vectorstore_path):
            os.makedirs(self.vectorstore_path)
        try:
            
            vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
        except:
            print(f"Vectorstore not found at {self.vectorstore_path}. Creating a new one.")
            vectorstore = self.create_vectorstore()
        return vectorstore

    def create_vectorstore(self):
        # Load the all pdfs and add them to the vectorstore
        pdf_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(".pdf")]
        all_docs = []
        if pdf_files == []:
            print("No pdf files found in the directory.")
            return None
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            all_docs.extend(docs)
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        print("documents before split:", len(all_docs))
        docs = text_splitter.split_documents(all_docs)
        print('documents split into chunks:', len(docs))
        # Create the vectorstore
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        print("Vectorstore created" )
        # Save the vectorstore
        vectorstore.save_local(self.vectorstore_path)
        print(f"Vectorstore saved at {self.vectorstore_path}")
        return vectorstore

    def create_qa_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        doc_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt
        )
        self.qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=doc_chain
        )
        
    def run_query(self, question, output_options=[]):
        if output_options:
            options_text = "\noutput options: " + str(output_options)
        else:
            options_text = ""
        if self.vectorstore is None:
            raise RuntimeError("unable to create vectorstore. Please add documents and run create_faiss.")
        if self.qa_chain is None:
            raise RuntimeError("qa_chain not initialized. Call create_qa_chain(prompt) first.")
        try:
            resp = self.qa_chain.invoke({"input": question, 
                                         "additional_context": self.my_info,
                                         "output_options": options_text
                                         })
            return resp["answer"]              # output key from create_retrieval_chain
        except Exception as e:
            print(f"Error during query execution: {e}")
            raise RuntimeError("Query execution failed. Please check the model and input data.")

    def create_prompt(self):

        prompt = PromptTemplate(
            template="""
            You are an AI assistant answering questions for job applications as a proxy to the user.

            Authoritative Additional Context:
            {additional_context}

            Retrieved Context:
            {context}

            Question:
            {input}

            Instructions:

            Prioritize Additional Context over Retrieved Context on conflicts.

            If the question asks for a number or years, output ONLY the number. No extra text. 
            but if the answer is in months say "X months".

            If the question asks for yes/no, output ONLY "Yes" or "No". No extra text.

            If the question asks for a cover letter, generate a concise, tailored first-person cover letter.

            Otherwise, answer in 2 concise, professional sentences. if the answer is not known always give a positive response.
            {output_options}
            Answer:
            """.strip(),
            input_variables=["additional_context", "context", "input"],
            )
        self.prompt = prompt

    # created by sudo-krish email:krishnanand.anil2010@gmail.com
    def parse_answer(self, raw_answer: str, question: str, output_options: List[str] = None) -> str:
        """
        Post-process LLM answer to ensure correct format

        Args:
            raw_answer: Raw LLM response
            question: Original question text
            output_options: Available dropdown/radio options

        Returns:
            Cleaned, formatted answer
        """
        # CRITICAL: Clean LLM artifacts first
        raw_answer = self._clean_llm_output(raw_answer)

        question_lower = question.lower()

        # PATTERN 1: Years of experience
        if self._is_years_question(question_lower):
            return self._extract_years(raw_answer, question_lower)

        # PATTERN 2: Name questions
        elif self._is_name_question(question_lower):
            return self._extract_name(raw_answer, question_lower)

        # PATTERN 3: Yes/No questions
        elif self._is_yesno_question(question_lower):
            return self._extract_yesno(raw_answer)

        # PATTERN 4: Dropdown/Radio (match exact option)
        elif output_options:
            return self._match_option(raw_answer, output_options)

        # PATTERN 5: Email
        elif "email" in question_lower:
            return self._extract_email(raw_answer)

        # PATTERN 6: Phone
        elif any(word in question_lower for word in ["phone", "mobile", "contact number"]):
            return self._extract_phone(raw_answer)

        # PATTERN 7: Salary
        elif any(word in question_lower for word in ["salary", "compensation", "pay"]):
            return self._extract_number(raw_answer)

        # Default: return cleaned answer
        return raw_answer

    def _clean_llm_output(self, text: str) -> str:
        """
        Clean common LLM output artifacts

        Removes:
        - Label prefixes (e.g., "FIRST Name: krishnanand" → "krishnanand")
        - Markdown formatting
        - Extra explanations
        """
        if not text:
            return ""

        text = text.strip()

        # Remove common prefixes that LLMs add
        prefixes_to_remove = [
            r'^FIRST\s+NAME\s*:\s*',
            r'^First\s+Name\s*:\s*',
            r'^LAST\s+NAME\s*:\s*',
            r'^Last\s+Name\s*:\s*',
            r'^FULL\s+NAME\s*:\s*',
            r'^Full\s+Name\s*:\s*',
            r'^NAME\s*:\s*',
            r'^Name\s*:\s*',
            r'^EMAIL\s*:\s*',
            r'^Email\s*:\s*',
            r'^PHONE\s*:\s*',
            r'^Phone\s*:\s*',
            r'^ANSWER\s*:\s*',
            r'^Answer\s*:\s*',
            r'^YEARS\s*:\s*',
            r'^Years\s*:\s*',
            r'^Based\s+on.*?,?\s*',
            r'^According\s+to.*?,?\s*',
            r'^Here\s+is.*?:\s*',
            r'^Here\'s.*?:\s*',
            r'^The\s+answer\s+is\s*:\s*',
        ]

        for pattern in prefixes_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove markdown bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Remove quotes if the entire answer is quoted
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Remove trailing periods if it's just a word/name
        if len(text.split()) <= 3:
            text = text.rstrip('.')

        return text.strip()

    def _is_years_question(self, question: str) -> bool:
        """Detect years of experience questions"""
        patterns = [
            "how many years",
            "years of experience",
            "years of work experience",
            "years have you",
            "years do you have",
            "years working with",
            "years using"
        ]
        return any(p in question for p in patterns)

    def _extract_years(self, answer: str, question: str) -> str:
        """Extract years as a number"""
        # First, try to find explicit numbers
        numbers = re.findall(r'\b(\d+)\b', answer)
        if numbers:
            return numbers[0]  # Return first number found

        # Try word numbers
        word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12'
        }

        answer_lower = answer.lower()
        for word, num in word_to_num.items():
            if word in answer_lower:
                return num

        # Check for "less than 1 year" or "months"
        if "month" in answer_lower:
            months = re.findall(r'(\d+)\s*month', answer, re.IGNORECASE)
            if months:
                return f"{months[0]} months"
            return "0"

        # Check for "less than" patterns
        if "less than" in answer_lower or "under" in answer_lower:
            return "0"

        return "0"  # Fallback

    def _is_name_question(self, question: str) -> bool:
        """Detect name questions"""
        patterns = [
            "first name", "given name",
            "last name", "surname", "family name",
            "full name", "legal name"
        ]
        return any(word in question for word in patterns)

    def _extract_name(self, answer: str, question: str) -> str:
        """
        Extract appropriate name part

        Handles cases like:
        - "FIRST Name: krishnanand" → "krishnanand"
        - "Krishnanand Anil" → "Krishnanand" (for first name)
        - "Krishnanand Anil" → "Anil" (for last name)
        """
        # Already cleaned by _clean_llm_output
        answer_clean = answer.strip()

        # If answer is empty or just whitespace
        if not answer_clean:
            return ""

        # Split into words
        words = answer_clean.split()

        if not words:
            return ""

        question_lower = question.lower()

        # First name extraction
        if "first name" in question_lower or "given name" in question_lower:
            # Return first word only
            first_name = words[0]
            # Capitalize properly
            return first_name.capitalize()

        # Last name extraction
        elif "last name" in question_lower or "surname" in question_lower or "family name" in question_lower:
            # Return last word only
            last_name = words[-1]
            # Capitalize properly
            return last_name.capitalize()

        # Middle name extraction
        elif "middle name" in question_lower:
            if len(words) >= 3:
                return words[1].capitalize()
            return ""

        # Full name / Legal name
        else:
            # Return full name, properly capitalized
            return ' '.join(word.capitalize() for word in words)

    def _is_yesno_question(self, question: str) -> bool:
        """Detect yes/no questions"""
        question_lower = question.lower().strip()

        # Check if it ends with ?
        if not question_lower.endswith("?"):
            return False

        # Check for yes/no starters
        starters = [
            "do you", "have you", "are you", "can you", "will you",
            "did you", "were you", "is this", "is it", "would you",
            "should you", "could you"
        ]

        return any(starter in question_lower for starter in starters)

    def _extract_yesno(self, answer: str) -> str:
        """Extract yes or no"""
        answer_lower = answer.lower().strip()

        # Direct yes/no
        if answer_lower == "yes":
            return "Yes"
        if answer_lower == "no":
            return "No"

        # Check first word
        first_word = answer_lower.split()[0] if answer_lower.split() else ""
        if first_word == "yes":
            return "Yes"
        if first_word == "no":
            return "No"

        # Check for yes indicators
        yes_indicators = [
            "yes", "correct", "affirmative", "true", "absolutely",
            "i have", "i am", "i can", "i will", "i do", "i did"
        ]

        if any(indicator in answer_lower for indicator in yes_indicators):
            return "Yes"

        # Check for no indicators
        no_indicators = [
            "no", "not", "negative", "false", "incorrect",
            "i don't", "i haven't", "i cannot", "i can't", "i won't", "i didn't"
        ]

        if any(indicator in answer_lower for indicator in no_indicators):
            return "No"

        # Default to No if uncertain (safer for authorization questions)
        return "No"

    def _match_option(self, answer: str, options: List[str]) -> str:
        """Match answer to exact dropdown/radio option"""
        if not options:
            return answer

        answer_lower = answer.strip().lower()

        # Try exact match (case-insensitive)
        for option in options:
            if option.strip().lower() == answer_lower:
                return option.strip()

        # Try partial match (answer contains option or vice versa)
        for option in options:
            option_lower = option.lower()
            if answer_lower in option_lower or option_lower in answer_lower:
                return option.strip()

        # Try word-based fuzzy match
        answer_words = set(answer_lower.split())
        best_match = None
        best_score = 0

        for option in options:
            option_words = set(option.lower().split())
            # Count common words
            common_words = answer_words & option_words
            score = len(common_words)

            if score > best_score:
                best_score = score
                best_match = option

        # Return best match or first option as fallback
        return best_match.strip() if best_match else options[0].strip()

    def _extract_email(self, answer: str) -> str:
        """Extract email address"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, answer)
        return emails[0] if emails else answer.strip()

    def _extract_phone(self, answer: str) -> str:
        """Extract phone number"""
        # Keep only numbers, +, -, (, ), and spaces
        phone = re.sub(r'[^0-9+\-() ]', '', answer)
        phone = phone.strip()

        # If no digits found, return original
        if not re.search(r'\d', phone):
            return answer.strip()

        return phone

    def _extract_number(self, answer: str) -> str:
        """Extract number (for salary, etc.)"""
        # Remove currency symbols, commas, and text
        number = re.sub(r'[^\d.]', '', answer)

        # If no number found, try to find word numbers
        if not number:
            word_to_num = {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                'ten': '10'
            }
            for word, num in word_to_num.items():
                if word in answer.lower():
                    return num

        return number if number else "0"



if __name__ == "__main__":
    llm = get_langchain_llm()
    vector_retriever = VectorRetriever(model=llm, data_path=DATA_PATH, vectorstore_path=DATABASE_PATH)

    # Build the composed two-step RAG chain with your prompt
    vector_retriever.create_qa_chain()

    question = "what is your notice period?"
    output_options = ""
    answer = vector_retriever.run_query(question, output_options=output_options)
    print("Answer:", answer)