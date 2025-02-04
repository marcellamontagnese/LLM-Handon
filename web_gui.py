import gradio as gr
from virtual_patient import VirtualPatient

patient = VirtualPatient()
patient.load_cases_from_file("100cases without cover and ending.txt")
#patient.load_cases_from_file("master_prompt.txt")

def respond(message, history):
    return patient.interact(message)

with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(respond)
    next_button = gr.Button("Next Case")
    
    def next_case():
        if len(patient.all_cases) > 1:
            patient.all_cases = patient.all_cases[1:]
            patient.set_case(patient.all_cases[0])
            return "New case started. How can I help you today?"
        return "No more cases available"
    
    next_button.click(fn=next_case, outputs=[chatbot.chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)