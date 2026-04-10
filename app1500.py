import os
import spaces
import gradio as gr
import numpy as np
from PIL import Image
import random
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
import torch
from transformers import pipeline as transformers_pipeline
import re
from cohere import ClientV2  # Changed from HuggingFace to Cohere

# ------------------------------------------------------------
# DEVICE SETUP
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# STABLE DIFFUSION XL PIPELINE (Text-to-Image)
# ------------------------------------------------------------
pipe = StableDiffusionXLPipeline.from_pretrained(
    "Heartsync/NSFW-Uncensored",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# Force important sub-modules to fp16 for VRAM efficiency
for sub in (pipe.text_encoder, pipe.text_encoder_2, pipe.vae, pipe.unet):
    sub.to(torch.float16)

# ------------------------------------------------------------
# STABLE DIFFUSION XL PIPELINE (Image-to-Image)
# ------------------------------------------------------------
# Мы просто берем компоненты (unet, vae, text_encoders) от первой модели (pipe), 
# чтобы не загружать их в память второй раз!
img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)

# ------------------------------------------------------------
# INITIALIZE COHERE CLIENT FOR TRANSLATIONS AND PROMPT GENERATION
# ------------------------------------------------------------
coh_api_key = os.getenv("COH_API")
if not coh_api_key:
    print("[WARNING] COH_API environment variable not found. LLM features will not work.")
    coh_client = None
else:
    try:
        coh_client = ClientV2(api_key=coh_api_key)
        print("[INFO] Cohere client initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Cohere client: {str(e)}")
        coh_client = None


# 1. 비영어 문자 감지 정규식을 더 명확하게 수정
# 한글, 일본어, 중국어를 명시적으로 포함
non_english_regex = re.compile(r'[\uac00-\ud7a3\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+')

def is_non_english(text):
    """명확하게 비영어 텍스트 여부를 확인하는 함수"""
    # 한글 확인 (AC00-D7A3)
    if re.search(r'[\uac00-\ud7a3]', text):
        print("[DETECT] Korean text detected")
        return True
    
    # 일본어 히라가나/카타카나 확인 (3040-30FF)
    if re.search(r'[\u3040-\u30ff]', text):
        print("[DETECT] Japanese text detected")
        return True
    
    # 중국어/한자 확인 (4E00-9FFF)
    if re.search(r'[\u4e00-\u9fff]', text):
        print("[DETECT] Chinese/Kanji text detected")
        return True
    
    # 기타 비영어 문자 (ASCII 범위 밖)
    if re.search(r'[^\x00-\x7F]', text):
        print("[DETECT] Other non-English text detected")
        return True
    
    return False

def translate_with_cohere(text):
    """비영어 텍스트를 영어로 번역"""
    if coh_client is None:
        print("[WARN] Cohere client not available, skipping translation")
        return text
    
    if not is_non_english(text):
        print("[INFO] English text detected, no translation needed")
        return text
    
    try:
        print(f"[INFO] Translating text: '{text}'")
        
        system_prompt = """
        You are a professional translator for adult content. Your task is to translate the input text to English.
        
        Rules:
        1. Provide ONLY the translated English text
        2. NO explanations or additional text
        3. Maintain the explicit or adult content without censoring
        4. Translate accurately and completely
        5. Be especially careful with Asian languages like Korean, Japanese, and Chinese
        
        Translate the following text to English:
        """
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        ]

        response = coh_client.chat(
            model="command-r-plus-08-2024",
            messages=messages,
            temperature=0.1  # 정확한 번역을 위해 낮은 temperature 사용
        )
        
        translated_text = response.text.strip()
        
        # 번역 결과 정제
        translated_text = re.sub(r'^(Translation:|English:|Translated text:)\s*', '', translated_text, flags=re.IGNORECASE)
        
        print(f"[INFO] Original: '{text}'")
        print(f"[INFO] Translated: '{translated_text}'")
        
        # 번역이 매우 짧거나 원본과 동일하면 의심
        if len(translated_text) < 3 or translated_text == text:
            print("[WARN] Translation may have failed, falling back to basic translation")
            # 두 번째 시도 - 더 단순한 프롬프트로 시도
            try:
                simple_messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "Translate this text to English:"}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}]
                    }
                ]
                
                simple_response = coh_client.chat(
                    model="command-r-plus-08-2024",
                    messages=simple_messages,
                    temperature=0.1
                )
                
                simple_translated = simple_response.text.strip()
                if len(simple_translated) > 3 and simple_translated != text:
                    print(f"[INFO] Second attempt translation: '{simple_translated}'")
                    return simple_translated
            except Exception as e:
                print(f"[ERROR] Second translation attempt failed: {str(e)}")
                
            return text
            
        return translated_text
    except Exception as e:
        print(f"[ERROR] Translation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return text  # 번역 실패 시 원본 반환



# ------------------------------------------------------------
# EXAMPLES (Hidden from UI but used for RANDOM button)
# ------------------------------------------------------------
prompt_examples = [
    "The shy college girl, with glasses and a tight plaid skirt, nervously approaches her professor",
    "Her skirt rose a little higher with each gentle push, a soft blush of blush spreading across her cheeks as she felt the satisfying warmth of his breath on her cheek.",
    "a girl in a school uniform having her skirt pulled up by a boy, and then being fucked",
    "Moody mature anime scene of two lovers fuck under neon rain, sensual atmosphere",
    "Moody mature anime scene of two lovers kissing under neon rain, sensual atmosphere",
    "The girl sits on the boy's lap by the window, his hands resting on her waist. She is unbuttoning his shirt, her expression focused and intense.",
    "A girl with long, black hair is sleeping on her desk in the classroom. Her skirt has ridden up, revealing her thighs, and a trail of drool escapes her slightly parted lips.",
    "The waves rolled gently, a slow, sweet kiss of the lip, a slow, slow build of anticipation as their toes bumped gently – a slow, sweet kiss of the lip, a promise of more to come.",
    "Her elegant silk gown swayed gracefully as she approached him, the delicate fabric brushing against her legs. A warm blush spread across her cheeks as she felt his breath on her face.",
    "Her white blouse and light cotton skirt rose a little higher with each gentle push, a soft blush spreading across her cheeks as she felt the satisfying warmth of his breath on her cheek.",
    "A woman in a business suit having her skirt lifted by a man, and then being sexually assaulted.",
    "The older woman sits on the man's lap by the fireplace, his hands resting on her hips. She is unbuttoning his vest, her expression focused and intense. He takes control of the situation as she finishes unbuttoning his shirt, pushing her onto her back and begins to have sex with her.",
    "There is a woman with long black hair. Her face features alluring eyes and full lips, with a slender figure adorned in black lace lingerie. She lies on the bed, loosening her lingerie strap with one hand while seductively glancing downward.",
    "In a dimly lit room, the same woman teases with her dark, flowing hair, now covering her voluptuous breasts, while a black garter belt accentuates her thighs. She sits on the sofa, leaning back, lifting one leg to expose her most private areas through the sheer lingerie.",
    "A woman with glasses, lying on the bed in just her bra, spreads her legs wide, revealing all! She wears a sultry expression, gazing directly at the viewer with her brown eyes, her short black hair cascading over the pillow. Her slim figure, accentuated by the lacy lingerie, exudes a seductive aura.",
    "A soft focus on the girl's face, eyes closed, biting her lip, as her roommate performs oral pleasure, the experienced woman's hair cascading between her thighs.",
    "A woman in a blue hanbok sits on a wooden floor, her legs folded beneath her, gazing out of a window, the sunlight highlighting the graceful lines of her clothing.",
    "The couple, immersed in a wooden outdoor bath, share an intimate moment, her wet kimono clinging to her curves, his hands exploring her body beneath the water's surface.",
    "A steamy shower scene, the twins embrace under the warm water, their soapy hands gliding over each other's curves, their passion intensifying as they explore uncharted territories.",
    "The teacher, with a firm grip, pins the student against the blackboard, her skirt hiked up, exposing her delicate lace panties. Their heavy breathing echoes in the quiet room as they share an intense, intimate moment.",
    "After hours, the girl sits on top of the teacher's lap, riding him on the classroom floor, her hair cascading over her face as she moves with increasing intensity, their bodies glistening with sweat.",
    "In the dimly lit dorm room, the roommates lay entangled in a passionate embrace, their naked bodies glistening with sweat, as the experienced woman teaches her lover the art of kissing and touching.",
    "The once-innocent student, now confident, takes charge, straddling her lover on the couch, their bare skin illuminated by the warm glow of the sunset through the window.",
    "A close-up of the secretary's hand unzipping her boss's dress shirt, her fingers gently caressing his chest, their eyes locked in a heated embrace in the supply closet.",
    "The secretary, in a tight pencil skirt and silk blouse, leans back on the boss's desk, her legs wrapped around his waist, her blouse unbuttoned, revealing her lace bra, as he passionately kisses her, his hands exploring her body.",
    "On the living room couch, one twin sits astride her sister's lap, their lips locked in a passionate kiss, their hands tangled in each other's hair, unraveling a new level of intimacy.",
    "In a dimly lit chamber, the dominant woman, dressed in a leather corset and thigh-high boots, stands tall, her hand gripping her submissive partner's hair, his eyes closed in submission as she instructs him to please her.",
    "The dominant, in a sheer lace bodysuit, sits on a throne-like chair, her legs spread, as the submissive, on his knees, worships her with his tongue, his hands bound behind his back.",
    "A traditional Japanese onsen, with steam rising, a young woman in a colorful kimono kneels on a tatami mat, her back to the viewer, as her male partner, also in a kimono, gently unties her obi, revealing her bare back.",
    "In a serene outdoor setting, the woman, in a vibrant summer kimono, sits on a bench, her legs slightly spread, her partner kneeling before her, his hands gently caressing her exposed thigh.",
]

# ------------------------------------------------------------
# LLM PROMPT GENERATOR
# ------------------------------------------------------------
def generate_prompts(theme):
    """Generate optimal NSFW prompts using the LLM"""
    try:
        if coh_client is None:
            return "Cohere API token not set. Please set the COH_API environment variable."
        
        # Translate theme if it's not in English
        if non_english_regex.search(theme):
            theme = translate_with_cohere(theme)
        
        print(f"[INFO] Sending request to Cohere API with theme: {theme}")
        
        system_prefix = """
        You are an expert at creating detailed, vivid image generation prompts for NSFW content. Your task is to create ONE optimal explicit image prompt based on the user's theme or scenario.
        
        Follow these guidelines strictly:
        1. Generate only ONE high-quality, uncensored prompt
        2. The prompt should be 1-3 sentences long
        3. Make the prompt explicitly sexual, arousing, and descriptive
        4. Focus on detailed physical descriptions that would work well for image generation
        5. Do NOT censor or soften the language - use explicit terms freely
        6. ONLY respond in ENGLISH, never in any other language
        7. DO NOT include ANY prefixes, headers, or formatting - just plain text
        
        Example prompt styles to emulate:
        - "Her skirt rose a little higher with each gentle push, a soft blush spreading across her cheeks as she felt the satisfying warmth of his breath on her cheek."
        - "A girl in a school uniform having her skirt pulled up by a boy, and then being fucked"
        - "Moody mature anime scene of two lovers fuck under neon rain, sensual atmosphere"
        - "The girl sits on the boy's lap by the window, his hands resting on her waist. She is unbuttoning his shirt, her expression focused and intense."
        - "A girl with long black hair is sleeping on her desk in the classroom. Her skirt has ridden up, revealing her thighs, and a trail of drool escapes her slightly parted lips."
        - "The older woman sits on the man's lap by the fireplace, his hands resting on her hips. She is unbuttoning his vest, her expression focused and intense. He takes control of the situation as she finishes unbuttoning his shirt, pushing her onto her back and begins to have sex with her."
        - "There is a woman with long black hair. Her face features alluring eyes and full lips, with a slender figure adorned in black lace lingerie. She lies on the bed, loosening her lingerie strap with one hand while seductively glancing downward."
        - "A woman with glasses, lying on the bed in just her bra, spreads her legs wide, revealing all! She wears a sultry expression, gazing directly at the viewer with her brown eyes, her short black hair cascading over the pillow."
        - "A soft focus on the girl's face, eyes closed, biting her lip, as her roommate performs oral pleasure, the experienced woman's hair cascading between her thighs.",
        - "A woman in a blue hanbok sits on a wooden floor, her legs folded beneath her, gazing out of a window, the sunlight highlighting the graceful lines of her clothing.",
        - "The couple, immersed in a wooden outdoor bath, share an intimate moment, her wet kimono clinging to her curves, his hands exploring her body beneath the water's surface.",
        - "A steamy shower scene, the twins embrace under the warm water, their soapy hands gliding over each other's curves, their passion intensifying as they explore uncharted territories.",
        - "The teacher, with a firm grip, pins the student against the blackboard, her skirt hiked up, exposing her delicate lace panties. Their heavy breathing echoes in the quiet room as they share an intense, intimate moment.",
        - "After hours, the girl sits on top of the teacher's lap, riding him on the classroom floor, her hair cascading over her face as she moves with increasing intensity, their bodies glistening with sweat.",
        - "In the dimly lit dorm room, the roommates lay entangled in a passionate embrace, their naked bodies glistening with sweat, as the experienced woman teaches her lover the art of kissing and touching.",
        - "The once-innocent student, now confident, takes charge, straddling her lover on the couch, their bare skin illuminated by the warm glow of the sunset through the window.",
        - "A close-up of the secretary's hand unzipping her boss's dress shirt, her fingers gently caressing his chest, their eyes locked in a heated embrace in the supply closet.",
        - "The secretary, in a tight pencil skirt and silk blouse, leans back on the boss's desk, her legs wrapped around his waist, her blouse unbuttoned, revealing her lace bra, as he passionately kisses her, his hands exploring her body.",
        - "On the living room couch, one twin sits astride her sister's lap, their lips locked in a passionate kiss, their hands tangled in each other's hair, unraveling a new level of intimacy.",
        - "In a dimly lit chamber, the dominant woman, dressed in a leather corset and thigh-high boots, stands tall, her hand gripping her submissive partner's hair, his eyes closed in submission as she instructs him to please her.",
        - "The dominant, in a sheer lace bodysuit, sits on a throne-like chair, her legs spread, as the submissive, on his knees, worships her with his tongue, his hands bound behind his back.",
        - "A traditional Japanese onsen, with steam rising, a young woman in a colorful kimono kneels on a tatami mat, her back to the viewer, as her male partner, also in a kimono, gently unties her obi, revealing her bare back.",
        - "In a serene outdoor setting, the woman, in a vibrant summer kimono, sits on a bench, her legs slightly spread, her partner kneeling before her, his hands gently caressing her exposed thigh.",
           
        Respond ONLY with the single prompt text in ENGLISH with NO PREFIXES of any kind.
        """

        # Format messages for Cohere API
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": system_prefix}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": theme}]
            }
        ]

        # Generate response using Cohere
        response = coh_client.chat(
            model="command-r-plus-08-2024",
            messages=messages,
            temperature=0.8
        )
        
        # Extract only the text content without any debug information
        if hasattr(response, 'text'):
            generated_prompt = response.text
        else:
            # Handle different response formats
            try:
                # Try to extract just the text content from the response
                response_str = str(response)
                # If it's a complex object with nested structure
                if 'text=' in response_str:
                    text_match = re.search(r"text=['\"]([^'\"]+)['\"]", response_str)
                    if text_match:
                        generated_prompt = text_match.group(1)
                    else:
                        generated_prompt = response_str
                else:
                    generated_prompt = response_str
            except:
                generated_prompt = str(response)
        
        # FORCE translation to English if there's any non-English content
        if non_english_regex.search(generated_prompt):
            print("[INFO] Translating non-English prompt to English")
            generated_prompt = translate_with_cohere(generated_prompt)
        
        # Clean the prompt
        generated_prompt = re.sub(r'^AI🐼:\s*', '', generated_prompt)
        generated_prompt = re.sub(r'^\d+[\.\)]\s*', '', generated_prompt)
        generated_prompt = re.sub(r'^(Prompt|Response|Result|Output):\s*', '', generated_prompt)
        generated_prompt = re.sub(r'^["\']+|["\']+$', '', generated_prompt)
        generated_prompt = generated_prompt.strip()
        generated_prompt = re.sub(r'\s+', ' ', generated_prompt)
        
        print(f"[INFO] Generated prompt: {generated_prompt}")
        
        # Final verification - check length and ensure it's English
        if len(generated_prompt) > 10:
            return generated_prompt
        else:
            return "Failed to generate a valid prompt"
        
    except Exception as e:
        print(f"[ERROR] Prompt generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating prompt: {str(e)}"


# ------------------------------------------------------------
# SDXL INFERENCE WRAPPER (Text-to-Image)
# ------------------------------------------------------------
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1500


@spaces.GPU(duration=60)  # 시간 제한 추가

def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    """
    중요: 프롬프트 텍스트에 한글이나 다른 비영어 문자가 있으면 반드시 영어로 번역해야 합니다.
    """
    print(f"[DEBUG] Original prompt received: '{prompt}'")
    print(f"[DEBUG] Original negative prompt received: '{negative_prompt}'")
    
    # 한글/비영어 감지 및 번역 (prompt)
    has_korean = bool(re.search(r'[\uac00-\ud7a3]', prompt))
    has_non_english = bool(re.search(r'[^\x00-\x7F]', prompt))
    
    if has_korean or has_non_english:
        print(f"[ALERT] 비영어 프롬프트 감지됨: '{prompt}'")
        
        # Cohere를 사용하여 직접 번역
        if coh_client:
            try:
                # 번역용 시스템 프롬프트
                trans_system = "You are a translator. Translate the following text to English accurately. Only provide the translation, no comments or explanations."
                
                # 번역 요청
                trans_response = coh_client.chat(
                    model="command-r-plus-08-2024",
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": trans_system}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                    temperature=0.1
                )
                
                # 응답 처리 - 다양한 속성 접근 방법 시도
                translated_prompt = None
                
                # 방법 1: response.text
                try:
                    if hasattr(trans_response, 'text'):
                        translated_prompt = trans_response.text
                        print("[DEBUG] 방법 1 (text 속성) 성공")
                except:
                    pass
                
                # 방법 2: response.response
                if translated_prompt is None:
                    try:
                        if hasattr(trans_response, 'response'):
                            translated_prompt = trans_response.response
                            print("[DEBUG] 방법 2 (response 속성) 성공")
                    except:
                        pass
                
                # 방법 3: response dictionary access
                if translated_prompt is None:
                    try:
                        # 응답이 dictionary인 경우
                        if isinstance(trans_response, dict) and 'text' in trans_response:
                            translated_prompt = trans_response['text']
                            print("[DEBUG] 방법 3 (dictionary access) 성공")
                    except:
                        pass
                
                # 방법 4: 문자열 변환 후 파싱
                if translated_prompt is None:
                    try:
                        response_str = str(trans_response)
                        print(f"[DEBUG] Response structure: {response_str[:200]}...")
                        
                        # text= 패턴 찾기
                        match = re.search(r"text=['\"](.*?)['\"]", response_str)
                        if match:
                            translated_prompt = match.group(1)
                            print("[DEBUG] 방법 4 (정규식 파싱) 성공")
                        
                        # content 패턴 찾기
                        if not translated_prompt and 'content=' in response_str:
                            match = re.search(r"content=['\"](.*?)['\"]", response_str)
                            if match:
                                translated_prompt = match.group(1)
                                print("[DEBUG] 방법 4.1 (content 정규식) 성공")
                    except Exception as parse_err:
                        print(f"[DEBUG] 정규식 파싱 오류: {parse_err}")
                
                # 최종 결과 확인
                if translated_prompt:
                    translated_prompt = translated_prompt.strip()
                    print(f"[SUCCESS] 번역됨: '{prompt}' -> '{translated_prompt}'")
                    prompt = translated_prompt
                else:
                    # 마지막 수단: 전체 응답 구조 로깅
                    print(f"[DEBUG] Full response type: {type(trans_response)}")
                    print(f"[DEBUG] Full response dir: {dir(trans_response)}")
                    print(f"[DEBUG] Could not extract translation, keeping original prompt")
            except Exception as e:
                print(f"[ERROR] 프롬프트 번역 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                # 번역 실패 시 원본 유지
    
    # 한글/비영어 감지 및 번역 (negative_prompt)
    has_korean = bool(re.search(r'[\uac00-\ud7a3]', negative_prompt))
    has_non_english = bool(re.search(r'[^\x00-\x7F]', negative_prompt))
    
    if has_korean or has_non_english:
        print(f"[ALERT] 비영어 네거티브 프롬프트 감지됨: '{negative_prompt}'")
        
        # Cohere를 사용하여 직접 번역 (위와 동일한 방식으로)
        if coh_client:
            try:
                trans_system = "You are a translator. Translate the following text to English accurately. Only provide the translation, no comments or explanations."
                
                trans_response = coh_client.chat(
                    model="command-r-plus-08-2024",
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": trans_system}]},
                        {"role": "user", "content": [{"type": "text", "text": negative_prompt}]}
                    ],
                    temperature=0.1
                )
                
                # 다양한 방법으로 응답 처리 (프롬프트 처리와 동일)
                translated_negative = None
                
                # 각종 접근 방법 (동일한 로직 적용)
                try:
                    if hasattr(trans_response, 'text'):
                        translated_negative = trans_response.text
                    elif hasattr(trans_response, 'response'):
                        translated_negative = trans_response.response
                    elif isinstance(trans_response, dict) and 'text' in trans_response:
                        translated_negative = trans_response['text']
                    else:
                        response_str = str(trans_response)
                        match = re.search(r"text=['\"](.*?)['\"]", response_str)
                        if match:
                            translated_negative = match.group(1)
                        elif 'content=' in response_str:
                            match = re.search(r"content=['\"](.*?)['\"]", response_str)
                            if match:
                                translated_negative = match.group(1)
                except Exception as parse_err:
                    print(f"[DEBUG] 네거티브 파싱 오류: {parse_err}")
                
                if translated_negative:
                    translated_negative = translated_negative.strip()
                    print(f"[SUCCESS] 네거티브 번역됨: '{negative_prompt}' -> '{translated_negative}'")
                    negative_prompt = translated_negative
            except Exception as e:
                print(f"[ERROR] 네거티브 프롬프트 번역 실패: {str(e)}")
    
    print(f"[INFO] 최종 사용될 프롬프트: '{prompt}'")
    print(f"[INFO] 최종 사용될 네거티브 프롬프트: '{negative_prompt}'")

    if len(prompt.split()) > 60:
        print("[WARN] Prompt >60 words — CLIP may truncate it.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)

    try:
        output_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        return output_image, seed
    except RuntimeError as e:
        print(f"[ERROR] Diffusion failed → {e}")
        return Image.new("RGB", (width, height), color=(0, 0, 0)), seed

# ------------------------------------------------------------
# SDXL INFERENCE WRAPPER (Image-to-Image)
# ------------------------------------------------------------
@spaces.GPU
def img2img_infer(init_image, prompt, negative_prompt, strength, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    """
    Image-to-Image generation function
    """
    if init_image is None:
        return None, seed
    
    print(f"[DEBUG] Image-to-Image prompt received: '{prompt}'")
    
    # 한글/비영어 감지 및 번역 (prompt)
    if is_non_english(prompt):
        print(f"[ALERT] 비영어 프롬프트 감지됨: '{prompt}'")
        prompt = translate_with_cohere(prompt)
        print(f"[INFO] 번역된 프롬프트: '{prompt}'")
    
    # 한글/비영어 감지 및 번역 (negative_prompt)
    if is_non_english(negative_prompt):
        print(f"[ALERT] 비영어 네거티브 프롬프트 감지됨: '{negative_prompt}'")
        negative_prompt = translate_with_cohere(negative_prompt)
        print(f"[INFO] 번역된 네거티브 프롬프트: '{negative_prompt}'")
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # 이미지 전처리
    init_image = init_image.convert("RGB")
    init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
    
    try:
        output_image = img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        return output_image, seed
    except RuntimeError as e:
        print(f"[ERROR] Image-to-Image generation failed → {e}")
        return None, seed

# Function to select a random example prompt
def get_random_prompt():
    return random.choice(prompt_examples)

# ------------------------------------------------------------
# UI LAYOUT + THEME (Enhanced Visual Design)
# ------------------------------------------------------------
css = """
body {background: linear-gradient(135deg, #f2e6ff 0%, #e6f0ff 100%); color: #222; font-family: 'Noto Sans', sans-serif;}
#col-container {margin: 0 auto; max-width: 768px; padding: 15px; background: rgba(255, 255, 255, 0.8); border-radius: 15px; box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);}
.gr-button {background: #7fbdf6; color: #fff; border-radius: 8px; transition: all 0.3s ease; font-weight: bold;}
.gr-button:hover {background: #5a9ae6; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.1);}
#prompt-box textarea {font-size: 1.1rem; height: 9rem !important; background: #fff; color: #222; border-radius: 10px; border: 1px solid #d1c1e0;}
.boost-btn {background: #ff7eb6; margin-top: 5px;}
.boost-btn:hover {background: #ff5aa5;}
.random-btn {background: #9966ff; margin-top: 5px;}
.random-btn:hover {background: #8040ff;}
.container {animation: fadeIn 0.5s ease-in-out;}
.title {color: #6600cc; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);}
.gr-form {border: none !important; background: transparent !important;}
.gr-input {border-radius: 8px !important;}
.gr-slider {height: 12px !important;}
.gr-slider .handle {height: 20px !important; width: 20px !important;}
.panel {border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1);}
.gr-image {border-radius: 12px; overflow: hidden; transition: all 0.3s ease;}
.gr-image:hover {transform: scale(1.02); box-shadow: 0 8px 25px rgba(0,0,0,0.15);}
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(20px);}
  to {opacity: 1; transform: translateY(0);}
}
.gr-accordion {border-radius: 10px; overflow: hidden; transition: all 0.3s ease;}
.gr-accordion:hover {box-shadow: 0 5px 15px rgba(0,0,0,0.1);}
"""


author_note = (
    "**ℹ️ A research platform pushing the limits of uncensored AI image generation. Input prompts in any language - automatic translation and generation supported.**"
    
)


# Function to boost prompt with LLM
def boost_prompt(keyword):
    if not keyword or keyword.strip() == "":
        return "Please enter a keyword or theme first"
    
    if coh_client is None:
        return "Cohere API token not set. Please set the COH_API environment variable."
    
    print(f"[INFO] Generating boosted prompt for keyword: {keyword}")
    prompt = generate_prompts(keyword)
    
    # Final verification that we're only returning valid content
    if isinstance(prompt, str) and len(prompt) > 10 and not prompt.startswith("Error") and not prompt.startswith("Failed"):
        return prompt.strip()
    else:
        return "Failed to generate a suitable prompt. Please try again with a different keyword."


with gr.Blocks(
    css=css, 
    theme=gr.themes.Soft(),
    head="""
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-GTFK201G22"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-GTFK201G22');
    </script>
    """
) as demo:
    gr.Markdown(
        f"""
        ## 🖌️ NSFW Uncensored Text & Imagery: AI Limits Explorer  
        
        **New Update: Image-to-Image functionality has been added as a new tab! Upload your images and experiment with various transformations.**
        
        {author_note}
        """, elem_classes=["title"]
    )
    

    with gr.Group(elem_classes="model-description"):
        gr.HTML("""
        <p>
        <strong>Models Use cases: </strong><br>
        </p>
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 10px; margin-bottom: 20px;">
 
            <a href="https://huggingface.co/spaces/Heartsync/FREE-NSFW-HUB" target="_blank">
                <img src="https://img.shields.io/static/v1?label=huggingface&message=FREE%20NSFW%20HUB&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>
      
            <a href="https://huggingface.co/spaces/Heartsync/PornHUB" target="_blank">
                <img src="https://img.shields.io/static/v1?label=Porn%20HUB&message=NSFW%20Uncensored&color=%23ffc0cb&labelColor=%23ffff00&logo=huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>
            <a href="https://huggingface.co/spaces/Heartsync/adult" target="_blank">
                <img src="https://img.shields.io/static/v1?label=Text%20to%20Image%20to%20Video&message=ADULT&color=%23ff00ff&labelColor=%23000080&logo=Huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>

            <a href="https://www.humangen.ai" target="_blank">
              <img src="https://img.shields.io/static/v1?label=100% FREE&message=AI%20Playground&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>   
        
            <a href="https://huggingface.co/spaces/Heartsync/NSFW-Uncensored-image" target="_blank">
                <img src="https://img.shields.io/static/v1?label=Image%20to%20Video&message=NSFW%20Uncensored&color=%230000ff&labelColor=%23800080&logo=Huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>            
            <a href="https://huggingface.co/spaces/Heartsync/NSFW-Uncensored-video2" target="_blank">
                <img src="https://img.shields.io/static/v1?label=Image%20to%20Video(Mirror)&message=NSFW%20Uncensored&color=%230000ff&labelColor=%23800080&logo=Huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>

            
        </div>
        <p>
        <small style="opacity: 0.8;">High-quality image generation powered by StableDiffusionXL with video generation capability. Supports long prompts and various artistic styles.</small>
        </p>
        """)
    
    # Create state variables to store the current image
    current_image = gr.State(None)
    current_seed = gr.State(0)

    # Tabs for Text-to-Image and Image-to-Image
    with gr.Tabs():
        # Text-to-Image Tab
        with gr.TabItem("Text to Image"):
            with gr.Column(elem_id="col-container", elem_classes=["container", "panel"]):
                # Add keyword input and boost button
                with gr.Row():
                    keyword_input = gr.Text(
                        label="Keyword Input",
                        show_label=True,
                        max_lines=1,
                        placeholder="Enter a keyword or theme in any language to generate an optimal prompt",
                        value="random",
                    )
                    boost_button = gr.Button("BOOST", elem_classes=["boost-btn"])
                    random_button = gr.Button("RANDOM", elem_classes=["random-btn"])
                
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        elem_id="prompt-box",
                        show_label=True,
                        max_lines=3,  # Increased to 3 lines (3x original)
                        placeholder="Enter your prompt in any language (Korean, English, Japanese, etc.)",
                    )
                    run_button = gr.Button("Generate", scale=0)

                # Image output area
                result = gr.Image(label="Generated Image", elem_classes=["gr-image"])

                with gr.Accordion("Advanced Settings", open=False, elem_classes=["gr-accordion"]):
                    negative_prompt = gr.Text(
                        label="Negative prompt",
                        max_lines=1,
                        placeholder="Enter a negative prompt in any language",
                        value="text, talk bubble, low quality, watermark, signature",
                    )

                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)
                        height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)

                    with gr.Row():
                        guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=20.0, step=0.1, value=7)
                        num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=28, step=1, value=28)

        # Image-to-Image Tab
        with gr.TabItem("Image to Image"):
            with gr.Column(elem_id="col-container", elem_classes=["container", "panel"]):
                # Input image
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    elem_classes=["gr-image"]
                )
                
                # Prompt input
                with gr.Row():
                    img2img_prompt = gr.Text(
                        label="Prompt",
                        show_label=True,
                        max_lines=3,
                        placeholder="Describe how you want to transform the image (any language)",
                    )
                    img2img_run_button = gr.Button("Transform", scale=0)
                
                # Output image
                img2img_result = gr.Image(label="Transformed Image", elem_classes=["gr-image"])
                
                # Image-to-Image advanced settings
                with gr.Accordion("Advanced Settings", open=False, elem_classes=["gr-accordion"]):
                    img2img_negative_prompt = gr.Text(
                        label="Negative prompt",
                        max_lines=1,
                        placeholder="What to avoid in the transformation",
                        value="low quality, watermark, signature",
                    )
                    
                    strength = gr.Slider(
                        label="Transformation Strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.75,
                        info="Lower values preserve more of the original image"
                    )
                    
                    img2img_seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    img2img_randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    with gr.Row():
                        img2img_width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)
                        img2img_height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)
                    
                    with gr.Row():
                        img2img_guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=20.0, step=0.1, value=7.5)
                        img2img_num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=30)

    # Define a function to store the generated image in state
    def update_image_state(img, seed_val):
        return img, seed_val

    # Connect boost button to generate prompt
    boost_button.click(
        fn=boost_prompt,
        inputs=[keyword_input],
        outputs=[prompt]
    )
    
    # Connect random button to insert random example
    random_button.click(
        fn=get_random_prompt,
        inputs=[],
        outputs=[prompt]
    )

    # Connect image generation button
    run_button.click(
        fn=infer,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, current_seed]
    ).then(
        fn=update_image_state,
        inputs=[result, current_seed],
        outputs=[current_image, current_seed]
    )

    # Connect Image-to-Image button
    img2img_run_button.click(
        fn=img2img_infer,
        inputs=[
            input_image,
            img2img_prompt,
            img2img_negative_prompt,
            strength,
            img2img_seed,
            img2img_randomize_seed,
            img2img_width,
            img2img_height,
            img2img_guidance_scale,
            img2img_num_inference_steps
        ],
        outputs=[img2img_result, img2img_seed]
    )

demo.queue().launch(share=True)