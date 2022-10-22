## Tortoise TTS Gradio Setup

1. Download / clone the repository.
2. Install the dependencies:
    `pip install -r requirements.txt`
3. In case you run into some error regarding some `libsnd` file, please try running this once:
    `conda install -c conda-forge librosa`
and then the `pip install -r requirements.txt`.
4. Compile Tortoise TTS, by running:
    ```
    python3 setup.py install
    ```
5. Run `app.py`. In the first run, it will automatically try to download the checkpoints, and store in cache for next run.