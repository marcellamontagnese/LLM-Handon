from openai import OpenAI
import dotenv
import os
from typing import List, Dict
import json
import re
from difflib import SequenceMatcher

class CaseLoader:
    @staticmethod
    def parse_case(case_text: str) -> Dict:
        """Parse a case into structured data, separating diagnosis from case details"""
        lines = case_text.split('\n')
        case_data = {
            'specialty': '',
            'presenting_complaint': '',
            'case_details': '',
            'diagnosis': '',
            'case_number': ''
        }
        
        # Extract case number and presenting complaint from first line
        if lines[0].startswith('Specialty:'):
            case_data['specialty'] = lines[0].replace('Specialty:', '').strip()
            title_line = lines[1]
        else:
            title_line = lines[0]
            
        case_match = re.match(r'Case (\d+):\s*(.+)', title_line)
        if case_match:
            case_data['case_number'] = case_match.group(1)
            case_data['presenting_complaint'] = case_match.group(2)
        
        # Find the diagnosis (usually at the end or after "Diagnosis:" marker)
        case_text = '\n'.join(lines)
        diagnosis_patterns = [
            r'Diagnosis:?\s*([^\n]+)',
            r'The diagnosis is:?\s*([^\n]+)',
            r'This (patient|case) demonstrates:?\s*([^\n]+)'
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, case_text, re.IGNORECASE)
            if match:
                case_data['diagnosis'] = match.group(1).strip()
                # Remove diagnosis from case details
                case_text = re.sub(pattern, '', case_text, flags=re.IGNORECASE)
                break
        
        # Store remaining text as case details
        case_data['case_details'] = case_text.strip()
        
        return case_data

    @staticmethod
    def load_cases_from_file(filename: str) -> List[Dict]:
        """Load and parse all cases from file"""
        try:
            with open(filename, 'r', encoding='latin1') as f:
                content = f.read()
                
            # Split by "Case" keyword followed by number
            cases = re.split(r'(?=Case\s+\d+:)', content)
            
            # Process each case
            processed_cases = []
            current_specialty = "Unknown"
            
            for case in cases:
                case = case.strip()
                if not case:
                    continue
                    
                # Check if this is a specialty header
                if case.isupper() and len(case) < 50:
                    current_specialty = case
                    continue
                    
                # Parse and add case if it's valid
                if case.startswith("Case"):
                    full_case = f"Specialty: {current_specialty}\n{case}"
                    case_data = CaseLoader.parse_case(full_case)
                    if case_data['case_details']:  # Only add if we have actual case content
                        processed_cases.append(case_data)
            
            print(f"Successfully loaded {len(processed_cases)} cases")
            return processed_cases
                    
        except Exception as e:
            raise ValueError(f"Error loading cases: {str(e)}")

class VirtualPatient:
    def __init__(self):
        dotenv.load_dotenv()
        self.client = OpenAI()
        self.conversation_history: List[Dict] = []
        self.all_cases: List[Dict] = []
        self.current_case: Dict = None
        self.diagnosis_made = False
        self.diagnosis_correct = False
        
        # Initialize system prompt template
        self.system_prompt_template = """You are simulating a patient case for medical training. Respond as the patient would, with appropriate emotions and lay terminology. Important rules:

1. Never reveal the diagnosis or medical details that the patient wouldn't know
2. Only provide information when specifically asked
3. Use natural, patient-like language (avoid medical terminology unless the patient would know it)
4. Maintain consistency with previous answers
5. Show appropriate emotions and concerns
6. If asked about a symptom or history not mentioned in the case, respond with "I don't think so" or "No"

Current presenting complaint: {presenting_complaint}

Remember: You are the patient. Respond naturally to the doctor's questions. Never reveal the diagnosis or medical details that a real patient wouldn't know."""

    def load_cases_from_file(self, filename: str):
        """Load all cases from a file"""
        self.all_cases = CaseLoader.load_cases_from_file(filename)
        if self.all_cases:
            self.set_case(self.all_cases[0])

    def set_case(self, case: Dict):
        """Set the current case and reset conversation"""
        self.current_case = case
        self.conversation_history = []
        self.diagnosis_made = False
        self.diagnosis_correct = False

    def interact(self, doctor_message: str) -> str:
        """Process a doctor's message and return the patient's response"""
        if not self.current_case:
            return "No case loaded. Please load a case first."

        # Check if this is a diagnosis attempt
        if any(keyword in doctor_message.lower() for keyword in ['my diagnosis is', 'i think this is', 'this could be', 'you have']):
            return self.check_diagnosis(doctor_message)

        # Add the doctor's message to the conversation history
        self.conversation_history.append({"role": "user", "content": doctor_message})
        
        # Create the system prompt with the current case details
        system_prompt = self.system_prompt_template.format(
            presenting_complaint=self.current_case['presenting_complaint']
        )

        # Add case details but keep them hidden from the prompt
        system_prompt += f"\n\nCase Details (for accurate responses but never reveal):\n{self.current_case['case_details']}"
        
        # Create the messages array for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history
        ]
        
        # Get response from OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        
        # Get the response content
        patient_response = response.choices[0].message.content
        
        # Add the response to conversation history
        self.conversation_history.append({"role": "assistant", "content": patient_response})
        
        return patient_response

    def check_diagnosis(self, doctor_message: str) -> str:
        """Check if the doctor's diagnosis matches the actual diagnosis"""
        self.diagnosis_made = True
        
        # Extract the diagnosis from the doctor's message
        # Remove common phrases used to introduce the diagnosis
        diagnosis_text = doctor_message.lower()
        for phrase in ['my diagnosis is', 'i think this is', 'this could be', 'you have', 'this is']:
            diagnosis_text = diagnosis_text.replace(phrase, '').strip()
        
        # Compare with actual diagnosis using fuzzy matching
        actual_diagnosis = self.current_case['diagnosis'].lower()
        similarity = SequenceMatcher(None, diagnosis_text, actual_diagnosis).ratio()
        
        self.diagnosis_correct = similarity > 0.8
        
        if self.diagnosis_correct:
            return (f"Correct! The diagnosis is {self.current_case['diagnosis']}. "
                   "Would you like to discuss the case or move to the next one?")
        else:
            return (f"That's not quite right. The actual diagnosis is {self.current_case['diagnosis']}. "
                   "Let's review the key points of the case. What made this diagnosis challenging?")

    def get_score(self) -> Dict:
        """Get the score and feedback for the current case"""
        if not self.diagnosis_made:
            return {
                "score": 0,
                "feedback": "No diagnosis attempted yet."
            }
        
        return {
            "score": 1 if self.diagnosis_correct else 0,
            "feedback": "Correct diagnosis!" if self.diagnosis_correct else "Incorrect diagnosis. Review the case details."
        }

# Example usage
if __name__ == "__main__":
    patient = VirtualPatient()
    
    # Load cases from file
    patient.load_cases_from_file("100cases without cover and ending.txt")
    
    print("Starting new case. Patient is ready for questions.")
    
    while True:
        doctor_input = input("\nDoctor: ")
        if doctor_input.lower() in ['quit', 'exit']:
            break
            
        response = patient.interact(doctor_input)
        print(f"\nPatient: {response}")
        
        if patient.diagnosis_made:
            score = patient.get_score()
            print(f"\nCase Score: {score['score']}")
            print(f"Feedback: {score['feedback']}")
            break