import gradio as gr
import torchaudio
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


def inference(text, voice, preset, seed):
    voice_samples, conditioning_latents = load_voice(voice)
    sample_voice = voice_samples[0] if len(voice_samples) else None
    gen = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=preset,
    )
    return (
        (22050, sample_voice.squeeze().cpu().numpy()),
        (24000, gen.squeeze().cpu().numpy()),
    )


def main():
    text = gr.Textbox(lines=4, label="Text:")
    preset = gr.Radio(
        ["ultra_fast", "fast", "standard", "high_quality"],
        value="fast",
        label="Preset mode (determines quality with tradeoff over speed):",
        type="value",
    )
    voice = gr.Dropdown(
        VOICE_OPTIONS, value="angie", label="Select voice:", type="value"
    )
    seed = gr.Number(value=0, precision=0, label="Seed (for reproducibility):")
    selected_voice = gr.Audio(label="Sample of selected voice:")
    output_audio = gr.Audio(label="Output:")

    interface = gr.Interface(
        fn=inference,
        inputs=[text, voice, preset, seed],
        outputs=[selected_voice, output_audio],
    )
    interface.launch(share=True)


if __name__ == "__main__":
    tts = TextToSpeech()
    main()
