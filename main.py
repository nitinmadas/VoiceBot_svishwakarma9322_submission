# # import gradio as gr
# # from modules.response_gen import agent
# # # from transcriber import transcribe_audio  # <-- Uncomment when ready
# # import uuid

# # # Unified processing function
# # def process_input(text):
# #     # if not audio and not text:
# #     #     return "Please provide either an audio recording or text."

# #     # if audio:
# #     #     try:
# #     #         user_input = text

# #     #         # user_input = transcribe_audio(audio)  # <-- Use this once your transcriber is ready
# #     #         # return "Audio input received, but transcription is not yet implemented."  # Placeholder
# #     #     except Exception as e:
# #     #         return f"Error in audio processing: {str(e)}"
# #     # else:
# #     #     user_input = text
    
# #     user_input = text

# #     thread_id = f"lenden-thread-{uuid.uuid4()}"
# #     config = {"configurable": {"thread_id": thread_id}}

# #     try:
# #         result = agent.invoke(
# #             {"messages": [{"role": "user", "content": user_input}]},
# #             config=config
# #         )
# #         response = result["messages"][-1].content
# #         return response
# #     except Exception as e:
# #         return f"Error from agent: {str(e)}"

# # # Build Gradio UI
# # with gr.Blocks() as demo:
# #     gr.Markdown("## 🎙️💬 Talk or Type with LendenClub AI Agent")

# #     with gr.Row():
# #         # audio_input = gr.Audio(source="microphone", type="filepath", label="🎤 Record Audio")
# #         text_input = gr.Textbox(lines=3, placeholder="Or type your message here...", label="💬 Text Input")

# #     submit_btn = gr.Button("Send to Agent")
# #     output = gr.Textbox(label="🧠 Agent Response")

# #     submit_btn.click(
# #         fn=process_input,
# #         inputs=[text_input],
# #         outputs=output
# #     )

# # # Launch it
# # if __name__ == "__main__":
# #     demo.launch()


# import gradio as gr
# from modules.response_gen import agent
# # from transcriber import transcribe_audio  # <-- Uncomment when ready
# import uuid

# # Unified processing function
# def process_chat(history, message):
#     thread_id = f"lenden-thread-{uuid.uuid4()}"
#     config = {"configurable": {"thread_id": thread_id}}

#     try:
#         result = agent.invoke(
#             {"messages": [{"role": "user", "content": message}]},
#             config=config
#         )
#         response = result["messages"][-1].content
#         history.append((message, response))
#         return history, ""
#     except Exception as e:
#         error_msg = f"Error from agent: {str(e)}"
#         history.append((message, error_msg))
#         return history, ""

# # Build Gradio Chatbot UI
# with gr.Blocks() as demo:
#     gr.Markdown("## 🤖 LendenClub AI Chatbot")

#     chatbot = gr.Chatbot(label="Chat with Lenden AI")
#     msg_input = gr.Textbox(placeholder="Type your message...", label="Your Message")
#     send_btn = gr.Button("Send")

#     send_btn.click(
#         fn=process_chat,
#         inputs=[chatbot, msg_input],
#         outputs=[chatbot, msg_input]
#     )

# # Launch
# if __name__ == "__main__":
#     demo.launch()

import gradio as gr
import uuid
import speech_recognition as sr
from modules.response_gen import agent  # Ensure this is implemented correctly


# === Speech Recognition Class ===
class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen_speech(self, language='en', timeout=7):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=timeout)

                if language == 'hi':
                    return self.recognizer.recognize_google(audio, language='hi-IN')
                return self.recognizer.recognize_google(audio, language='en-US')
        except sr.WaitTimeoutError:
            return "No speech detected. Please try speaking again."
        except sr.UnknownValueError:
            return "Could not understand audio. Please speak more clearly."
        except sr.RequestError as e:
            return f"Error connecting to speech recognition service: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"


# === Agent Handler ===
def process_text_input(history, message):
    thread_id = f"lenden-thread-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )
        response = result["messages"][-1].content
        history.append((message, response))
        return history, ""
    except Exception as e:
        history.append((message, f"Error from agent: {str(e)}"))
        return history, ""


# === Voice Handler that mimics user input ===
stt = SpeechToText()

def handle_voice_input(lang, history):
    yield history, "🎙️ Listening..."

    lang_code = "en" if lang == "English" else "hi"
    text = stt.listen_speech(language=lang_code)

    if text.startswith(("No speech", "Could not", "Error", "Unexpected")):
        yield history, text
    else:
        # Voice input is treated like normal text input
        updated_history, _ = process_text_input(history, text)
        yield updated_history, ""


# === Gradio UI ===
with gr.Blocks(title="LendenClub Chatbot (Text + Voice)") as demo:
    gr.Markdown("## 🤖 LendenClub AI Chatbot (Talk or Type)")
    
    chatbot = gr.Chatbot(label="Chat with Lenden AI", height=400)
    msg_input = gr.Textbox(placeholder="Type your message...", label="💬 Message")

    with gr.Row():
        submit_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Chat")

    with gr.Row():
        language_selection = gr.Dropdown(
            choices=["English", "Hindi"],
            label="🎙️ Select Voice Language",
            value="English"
        )
        voice_status = gr.Textbox(label="Voice Input Status", interactive=False)
        voice_btn = gr.Button("🎤 Speak", variant="primary")

    # --- Text Submit Handlers ---
    submit_btn.click(
        fn=process_text_input,
        inputs=[chatbot, msg_input],
        outputs=[chatbot, msg_input]
    )

    msg_input.submit(
        fn=process_text_input,
        inputs=[chatbot, msg_input],
        outputs=[chatbot, msg_input]
    )

    # --- Voice Button Handler ---
    voice_btn.click(
        fn=handle_voice_input,
        inputs=[language_selection, chatbot],
        outputs=[chatbot, voice_status],
        show_progress="hidden"
    )

    # --- Clear Chat ---
    clear_btn.click(lambda: [], outputs=[chatbot])

# Launch the app
if __name__ == "__main__":
    demo.launch()
