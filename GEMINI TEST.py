import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile

# Replace AIzaSyBY88kNHfPUqauW2z5wu5-qnBv1Kr4d86s'YOUR_API_KEY' with your actual Gemini API key
API_KEY = "AIzaSyBY88kNHfPUqauW2z5wu5-qnBv1Kr4d86s"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def get_gemini_response(prompt):
    """
    Sends a prompt to the Gemini API and returns the response.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "I'm sorry, I couldn't process your request."

def speak(text):
    """Converts text to speech and plays the audio."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)
            sound = AudioSegment.from_mp3(temp_path)
            play(sound)
    except Exception as e:
        print(f"Speech output error: {e}")

def main():
    """Main loop for the voice-enabled Gemini chatbot."""
    while True:
        user_input = input("Enter your prompt (or 'exit' to quit): ")

        if user_input.lower() in ["exit", "quit"]:
            speak("Goodbye! Have a nice day.")
            break

        if user_input:
            response_text = get_gemini_response(user_input)
            print("Gemini Response:")
            print(response_text)
            speak(response_text)

if __name__ == "__main__":
    main()

