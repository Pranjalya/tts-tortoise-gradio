import gradio as gr
import torchaudio
import time
from datetime import datetime
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

VOICE_OPTIONS = [
    "angie",
    "cond_latent_example",
    "deniro",
    "freeman",
    "halle",
    "lj",
    "myself",
    "pat2",
    "snakes",
    "tom",
    "train_daws",
    "train_dreams",
    "train_grace",
    "train_lescault",
    "weaver",
    "applejack",
    "daniel",
    "emma",
    "geralt",
    "jlaw",
    "mol",
    "pat",
    "rainbow",
    "tim_reynolds",
    "train_atkins",
    "train_dotrice",
    "train_empire",
    "train_kennard",
    "train_mouse",
    "william",
    "random",  # special option for random voice
    "custom_voice",  # special option for custom voice
    "disabled",  # special option for disabled voice
]


def inference(text, emotion, prompt, voice, mic_audio, voice_b, voice_c, preset, seed):
    if voice != "custom_voice":
        voices = [voice]
    else:
        voices = []

    if voice_b != "disabled":
        voices.append(voice_b)
    if voice_c != "disabled":
        voices.append(voice_c)

    if emotion != "None/Custom":
        text = f"[I am really {emotion.lower()},] {text}"
    elif prompt.strip() != "":
        text = f"[{prompt},] {text}"

    c = None
    if voice == "custom_voice":
        if mic_audio is None:
            raise gr.Error("Please provide audio from mic when choosing custom voice")
        c = load_audio(mic_audio, 22050)

    if len(voices) == 1:
        if voice == "custom_voice":
            voice_samples, conditioning_latents = c, None
        else:
            voice_samples, conditioning_latents = load_voice(voice)
    else:
        voice_samples, conditioning_latents = load_voices(voices)
        if voice == "custom_voice":
            voice_samples.extend(c)

    sample_voice = voice_samples[0] if len(voice_samples) else None

    start_time = time.time()
    gen, _ = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=preset,
        use_deterministic_seed=seed,
        return_deterministic_state=True,
        k=1,
    )

    with open("Tortoise_TTS_Runs.log", "a") as f:
        f.write(
            f"{datetime.now()} | Voice: {','.join(voices)} | Text: {text} | Quality: {preset} | Time Taken (s): {time.time()-start_time} | Seed: {seed}\n"
        )

    return (
        (22050, sample_voice.squeeze().cpu().numpy()),
        (24000, gen[0].squeeze().cpu().numpy()),
        None, None
        # (24000, gen[1].squeeze().cpu().numpy()),
        # (24000, gen[2].squeeze().cpu().numpy()),
    )


def main():
    text = gr.Textbox(lines=4, label="Text:")
    emotion = gr.Radio(
        ["None/Custom", "Happy", "Sad", "Angry", "Disgusted", "Arrogant"],
        value="None/Custom",
        label="Select emotion:",
        type="value",
    )
    prompt = gr.Textbox(lines=1, label="Enter prompt if [Custom] emotion:")
    preset = gr.Radio(
        ["ultra_fast", "fast", "standard", "high_quality"],
        value="fast",
        label="Preset mode (determines quality with tradeoff over speed):",
        type="value",
    )
    voice = gr.Dropdown(
        VOICE_OPTIONS, value="angie", label="Select voice:", type="value"
    )
    mic_audio = gr.Audio(
        label="Record voice (when selected custom_voice):", source="microphone", type="filepath"
    )
    voice_b = gr.Dropdown(
        VOICE_OPTIONS,
        value="disabled",
        label="(Optional) Select second voice:",
        type="value",
    )
    voice_c = gr.Dropdown(
        VOICE_OPTIONS,
        value="disabled",
        label="(Optional) Select third voice:",
        type="value",
    )
    seed = gr.Number(value=0, precision=0, label="Seed (for reproducibility):")

    selected_voice = gr.Audio(label="Sample of selected voice (first):")
    output_audio_1 = gr.Audio(label="Output [Candidate 1]:")
    output_audio_2 = gr.Audio(label="Output [Candidate 2]:")
    output_audio_3 = gr.Audio(label="Output [Candidate 3]:")

    interface = gr.Interface(
        fn=inference,
        inputs=[text, emotion, prompt, voice, mic_audio, voice_b, voice_c, preset, seed],
        outputs=[selected_voice, output_audio_1, output_audio_2, output_audio_3],
    )
    interface.launch(share=True)


if __name__ == "__main__":
    tts = TextToSpeech()

    with open("Tortoise_TTS_Runs.log", "a") as f:
        f.write(
            f"\n\n-------------------------Tortoise TTS Logs, {datetime.now()}-------------------------\n"
        )

    main()
