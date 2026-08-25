import streamlit as st
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    AutoModelForMultimodalLM,
)
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import torch
import pyttsx3
import tempfile
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Image Understanding",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Understanding & Caption Generator")
st.write(
    "Upload an image and use AI to generate captions, detect objects, "
    "ask questions, create social media content and more."
)


# =========================================================
# LOAD BLIP CAPTION MODEL
# =========================================================

@st.cache_resource
def load_caption_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model.eval()

    return processor, model


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():

    model = YOLO("yolo11n.pt")

    return model


# =========================================================
# CAPTION GENERATION
# =========================================================

def generate_caption(image, style, processor, model):

    prompts = {

        "Simple":
            "a simple description of",

        "Descriptive":
            "a descriptive caption of",

        "Detailed":
            "a detailed description of",

        "Funny":
            "a funny description of",

        "Professional":
            "a professional description of",

        "Social Media":
            "a social media caption describing"
    }

    prompt = prompts.get(
        style,
        "a description of"
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=5
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


# =========================================================
# OBJECT DETECTION
# =========================================================

def detect_objects(image):

    yolo = load_yolo_model()

    results = yolo(
        image,
        verbose=False
    )

    detected_objects = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            object_name = result.names[class_id]

            detected_objects.append(
                {
                    "name": object_name,
                    "confidence": confidence
                }
            )

    return detected_objects


# =========================================================
# SOCIAL MEDIA CONTENT
# =========================================================

def generate_social_content(caption, objects):

    hashtags = []

    for obj in objects:

        tag = obj["name"].replace(
            " ",
            ""
        )

        hashtags.append(
            "#" + tag
        )

    hashtags.extend(
        [
            "#AI",
            "#ImageCaption",
            "#Photography"
        ]
    )

    # Remove duplicates
    hashtags = list(
        dict.fromkeys(hashtags)
    )

    hashtag_text = " ".join(
        hashtags
    )

    social_caption = (
        f"{caption} ✨\n\n"
        f"{hashtag_text}"
    )

    return social_caption, hashtag_text


# =========================================================
# TEXT TO SPEECH
# =========================================================

def text_to_speech(text):

    engine = pyttsx3.init()

    engine.setProperty(
        "rate",
        150
    )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    temp_file.close()

    engine.save_to_file(
        text,
        temp_file.name
    )

    engine.runAndWait()

    engine.stop()

    return temp_file.name




# =========================================================
# LOAD SMOLVLM
# =========================================================

@st.cache_resource
def load_vlm():

    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id)

    model.eval()

    return processor, model


# =========================================================
# SMOLVLM VISUAL UNDERSTANDING
# =========================================================

def run_vlm(image, question, max_new_tokens=200):

    processor, model = load_vlm()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[1]

    return text.strip()


def generate_detailed_description(image):

    return run_vlm(
        image,
        (
            "Describe this image in detail. Mention the main subjects, "
            "their actions or positions, important objects, foreground, "
            "background, environment, spatial relationships, and the "
            "overall scene. Write a coherent detailed description."
        ),
        max_new_tokens=280,
    )


def answer_image_question(image, question):

    return run_vlm(
        image,
        (
            "Answer the user's question using only information visible "
            "in the image. Be direct and concise. If the answer cannot "
            "be determined from the image, say so.\n\n"
            f"Question: {question}"
        ),
        max_new_tokens=150,
    )


def generate_scene_understanding(image):

    return run_vlm(
        image,
        (
            "Analyze the scene in this image. Identify the likely scene "
            "type, environment, main activity, and important elements. "
            "Present the result as short labeled lines."
        ),
        max_new_tokens=180,
    )


def generate_simple_caption(image):

    processor, model = load_caption_model()

    inputs = processor(
        images=image,
        text="a simple description of",
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
        )

    return processor.decode(
        output[0],
        skip_special_tokens=True,
    ).strip()


def generate_social_content_v2(image, platform, tone, objects):

    description = generate_detailed_description(image)

    object_names = list(
        dict.fromkeys(
            obj["name"].title() for obj in objects
        )
    )

    object_context = (
        " Visible elements include "
        + ", ".join(object_names[:6])
        + "."
        if object_names
        else ""
    )

    openers = {
        "Creative": "A moment worth capturing and sharing. ✨",
        "Funny": "This image definitely understood the assignment! 😄",
        "Professional": "A visually meaningful moment captured with clarity.",
        "Inspirational": "Sometimes one image can tell an entire story. 🌟",
        "Casual": "Just one of those moments worth sharing. ✨",
    }

    opener = openers.get(tone, openers["Creative"])

    if platform == "LinkedIn":
        cta = "What do you notice first in this image?"
    elif platform == "X":
        cta = "What story do you see here? 👀"
    elif platform == "Facebook":
        cta = "What do you think about this scene? ❤️"
    else:
        cta = "What story do you see in this image? 👇"

    caption = f"{opener}\n\n{description}{object_context}\n\n{cta}"

    tags = []
    for obj in objects:
        word = "".join(
            ch for ch in obj["name"] if ch.isalnum()
        )
        if word:
            tags.append("#" + word)

    platform_tags = {
        "Instagram": ["#AI", "#Photography", "#VisualStory"],
        "LinkedIn": ["#ArtificialIntelligence", "#ComputerVision", "#AI"],
        "Facebook": ["#AI", "#Photography", "#Amazing"],
        "X": ["#AI", "#ComputerVision", "#ImageAI"],
    }

    tags.extend(platform_tags.get(platform, ["#AI", "#Photography"]))
    tags = list(dict.fromkeys(tags))[:15]

    return caption, " ".join(tags)


# =========================================================
# TRANSLATION
# =========================================================

LANGUAGES = {
    "English": "en",
    "Kannada": "kn",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}


def translate_text(text, language_name):

    return GoogleTranslator(
        source="auto",
        target=LANGUAGES[language_name],
    ).translate(text)


def show_output_tools(text, key_prefix, tts_label="Listen"):

    st.code(text, language=None)

    st.markdown("### 🌐 Multilingual")

    language = st.selectbox(
        "Translate to:",
        list(LANGUAGES.keys()),
        key=f"{key_prefix}_language",
    )

    if st.button(
        "🌐 Translate",
        key=f"{key_prefix}_translate",
    ):

        with st.spinner("Translating..."):

            try:
                translated = translate_text(
                    text,
                    language,
                )

                st.session_state[
                    f"{key_prefix}_translated"
                ] = translated

            except Exception as e:
                st.error(f"Translation failed: {e}")

    translated_key = f"{key_prefix}_translated"

    if translated_key in st.session_state:

        st.markdown("#### 🌐 Translation")

        st.code(
            st.session_state[translated_key],
            language=None,
        )

    st.markdown("### 🔊 Text to Speech")

    if st.button(
        f"🔊 {tts_label}",
        key=f"{key_prefix}_tts",
    ):

        with st.spinner("Creating speech..."):

            try:
                audio_file = text_to_speech(text)

                with open(audio_file, "rb") as audio:
                    audio_bytes = audio.read()

                st.audio(
                    audio_bytes,
                    format="audio/wav",
                )

                try:
                    os.remove(audio_file)
                except OSError:
                    pass

            except Exception as e:
                st.error(
                    f"Text-to-speech failed: {e}"
                )



# =========================================================
# IMAGE UPLOAD + MAIN 3-TAB INTERFACE
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image to begin",
    type=["jpg", "jpeg", "png", "jfif", "webp"],
)

if uploaded_file is None:

    st.info(
        "👆 Upload an image above to unlock "
        "Caption Generator, Image Analysis and Ask About Image."
    )

else:

    image = Image.open(uploaded_file).convert("RGB")

    image_key = (
        uploaded_file.name,
        uploaded_file.size,
    )

    if st.session_state.get("image_key") != image_key:

        st.session_state["image_key"] = image_key

        for key in [
            "caption",
            "detailed_description",
            "objects",
            "annotated_image",
            "scene_understanding",
            "qa_answer",
            "social_caption",
            "hashtags",
            "translated",
            "qa_translated",
        ]:
            st.session_state.pop(key, None)

    st.subheader("🖼️ Uploaded Image")
    st.image(
        image,
        use_container_width=True,
    )

    # =====================================================
    # ONLY THREE MAIN TABS
    # =====================================================

    caption_tab, analysis_tab, qa_tab = st.tabs(
        [
            "📝 Caption Generator",
            "🔍 Image Analysis",
            "❓ Ask About Image",
        ]
    )

    # =====================================================
    # 1. CAPTION GENERATOR
    # =====================================================

    with caption_tab:

        st.header("📝 Caption Generator")

        purpose = st.radio(
            "Choose a caption purpose:",
            [
                "Simple",
                "Detailed / Descriptive",
                "Social Media",
            ],
            horizontal=True,
            key="caption_purpose",
        )

        st.divider()

        # -------------------------------------------------
        # SIMPLE
        # -------------------------------------------------

        if purpose == "Simple":

            st.subheader("✨ Simple Caption")

            if st.button(
                "✨ Generate Simple Caption",
                type="primary",
                key="generate_simple",
            ):

                with st.spinner(
                    "Generating simple caption..."
                ):

                    try:
                        caption = generate_simple_caption(image)
                        st.session_state["caption"] = caption

                    except Exception as e:
                        st.error(
                            f"Caption generation failed: {e}"
                        )

            if "caption" in st.session_state:

                st.success("Caption generated.")

                st.write(
                    st.session_state["caption"]
                )

                show_output_tools(
                    st.session_state["caption"],
                    "simple",
                    "Listen to Caption",
                )

        # -------------------------------------------------
        # DETAILED / DESCRIPTIVE
        # -------------------------------------------------

        elif purpose == "Detailed / Descriptive":

            st.subheader("📖 Detailed / Descriptive")

            if st.button(
                "🧠 Generate Detailed Description",
                type="primary",
                key="generate_detailed",
            ):

                with st.spinner(
                    "Analyzing the image in detail..."
                ):

                    try:
                        description = (
                            generate_detailed_description(
                                image
                            )
                        )

                        st.session_state[
                            "detailed_description"
                        ] = description

                    except Exception as e:
                        st.error(
                            f"Detailed description failed: {e}"
                        )

            if "detailed_description" in st.session_state:

                st.write(
                    st.session_state[
                        "detailed_description"
                    ]
                )

                show_output_tools(
                    st.session_state[
                        "detailed_description"
                    ],
                    "detailed",
                    "Listen to Description",
                )

        # -------------------------------------------------
        # SOCIAL MEDIA
        # -------------------------------------------------

        else:

            st.subheader("📱 Social Media")

            col1, col2 = st.columns(2)

            with col1:

                platform = st.selectbox(
                    "Platform",
                    [
                        "Instagram",
                        "LinkedIn",
                        "Facebook",
                        "X",
                    ],
                    key="social_platform",
                )

            with col2:

                tone = st.selectbox(
                    "Tone",
                    [
                        "Creative",
                        "Funny",
                        "Professional",
                        "Inspirational",
                        "Casual",
                    ],
                    key="social_tone",
                )

            if st.button(
                "📱 Generate Social Media Content",
                type="primary",
                key="generate_social",
            ):

                with st.spinner(
                    "Creating social media content..."
                ):

                    try:

                        objects = st.session_state.get(
                            "objects",
                            [],
                        )

                        caption, hashtags = (
                            generate_social_content_v2(
                                image,
                                platform,
                                tone,
                                objects,
                            )
                        )

                        st.session_state[
                            "social_caption"
                        ] = caption

                        st.session_state[
                            "hashtags"
                        ] = hashtags

                    except Exception as e:
                        st.error(
                            f"Social media generation failed: {e}"
                        )

            if "social_caption" in st.session_state:

                st.markdown("### 📱 Caption")

                st.write(
                    st.session_state[
                        "social_caption"
                    ]
                )

                st.code(
                    st.session_state[
                        "social_caption"
                    ],
                    language=None,
                )

                st.markdown("### #️⃣ Hashtags")

                st.write(
                    st.session_state[
                        "hashtags"
                    ]
                )

                st.code(
                    st.session_state[
                        "hashtags"
                    ],
                    language=None,
                )

                st.caption(
                    "Use the copy control on each result above "
                    "to copy the caption or hashtags."
                )

    # =====================================================
    # 2. IMAGE ANALYSIS
    # =====================================================

    with analysis_tab:

        st.header("🔍 Image Analysis")

        st.subheader("🎯 Object Detection")

        confidence = st.slider(
            "Minimum confidence score",
            0.10,
            0.90,
            0.25,
            0.05,
            key="confidence_threshold",
        )

        if st.button(
            "🔎 Detect Objects",
            type="primary",
            key="detect_objects",
        ):

            with st.spinner(
                "Detecting objects..."
            ):

                try:

                    yolo = load_yolo_model()

                    results = yolo(
                        image,
                        conf=confidence,
                        verbose=False,
                    )

                    objects = []

                    for result in results:

                        if result.boxes is None:
                            continue

                        for box in result.boxes:

                            class_id = int(
                                box.cls[0].item()
                            )

                            score = float(
                                box.conf[0].item()
                            )

                            objects.append(
                                {
                                    "name": result.names[class_id],
                                    "confidence": score,
                                }
                            )

                    st.session_state[
                        "objects"
                    ] = objects

                    if results:

                        plotted = results[0].plot()

                        st.session_state[
                            "annotated_image"
                        ] = Image.fromarray(
                            plotted[:, :, ::-1]
                        )

                except Exception as e:
                    st.error(
                        f"Object detection failed: {e}"
                    )

        if "objects" in st.session_state:

            objects = st.session_state["objects"]

            if "annotated_image" in st.session_state:

                st.image(
                    st.session_state[
                        "annotated_image"
                    ],
                    caption="Detected objects",
                    use_container_width=True,
                )

            if objects:

                st.markdown("### 🎯 Objects + Confidence")

                for index, obj in enumerate(
                    objects,
                    start=1,
                ):

                    col1, col2, col3 = st.columns(
                        [0.15, 0.55, 0.30]
                    )

                    with col1:
                        st.write(f"**{index}**")

                    with col2:
                        st.write(
                            obj["name"].title()
                        )

                    with col3:
                        st.write(
                            f"**{obj['confidence'] * 100:.1f}%**"
                        )

            else:

                st.info(
                    "No objects were detected at the selected confidence level."
                )

        st.divider()

        st.subheader("🌎 Scene Understanding")

        if st.button(
            "🌎 Analyze Scene",
            key="analyze_scene",
        ):

            with st.spinner(
                "Understanding the scene..."
            ):

                try:
                    scene = generate_scene_understanding(
                        image
                    )

                    st.session_state[
                        "scene_understanding"
                    ] = scene

                except Exception as e:
                    st.error(
                        f"Scene analysis failed: {e}"
                    )

        if "scene_understanding" in st.session_state:

            st.write(
                st.session_state[
                    "scene_understanding"
                ]
            )

            st.code(
                st.session_state[
                    "scene_understanding"
                ],
                language=None,
            )

    # =====================================================
    # 3. ASK ABOUT IMAGE
    # =====================================================

    with qa_tab:

        st.header("❓ Ask About Image")

        st.write(
            "Ask a question about the uploaded image."
        )

        question = st.text_area(
            "Your question",
            placeholder=(
                "Example: What objects are visible?\n"
                "Example: What is the main subject doing?\n"
                "Example: What is visible in the background?"
            ),
            height=100,
            key="image_question",
        )

        if st.button(
            "💬 Ask AI",
            type="primary",
            key="ask_ai",
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question first."
                )

            else:

                with st.spinner(
                    "Analyzing the image..."
                ):

                    try:

                        answer = answer_image_question(
                            image,
                            question.strip(),
                        )

                        st.session_state[
                            "qa_answer"
                        ] = answer

                    except Exception as e:
                        st.error(
                            f"Image Q&A failed: {e}"
                        )

        if "qa_answer" in st.session_state:

            st.divider()

            st.subheader("🤖 Answer")

            st.write(
                st.session_state[
                    "qa_answer"
                ]
            )

            show_output_tools(
                st.session_state[
                    "qa_answer"
                ],
                "qa",
                "Listen to Answer",
            )
