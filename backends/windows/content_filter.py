import re
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
class ContentFilter:
    def __init__(self):
        # Basic list of inappropriate terms
        self.inappropriate_terms = {
            # Profanity
            "profanity": [
                "ass", "asshole", "bastard", "bitch", "crap", "cock", "cunt", 
                "dick", "dildo", "fuck", "fucking", "fucker", "fuk", 
                "shit", "shitty", "tits", "twat", "wank", "whore"
            ],
            
            # Anatomical terms
            "anatomical": [
                "anus", "areola", "ass", "boob", "boobs", "booty", "butt", 
                "cock", "clit", "nipple", "penis", "pussy", "vagina", "vulva",
                "breast", "breasts", "cleavage", "genitals", "genitalia",
                "curves", "thighs"
            ],
            
            # Sexual activities
            "sexual_acts": [
                "anal", "blow", "bdsm", "cum", "climax", "dildo", "fuck",
                "fucking", "handjob", "hump", "jerk", "masturbate", "nude",
                "orgy", "orgasm", "porn", "rape", "sex", "slut", "suck", "xxx",
                "fornicate", "penetrate", "penetration", "foreplay", "erotic",
                "sexually", "aroused", "arousal", "fetish", "kink", "kinky"
            ],
            
            # Violence
            "violence": [
                "abuse", "assault", "attack", "beat", "blood", "brutal", 
                "death", "decapitate", "die", "gun", "hurt", "kill", "killing", 
                "murder", "mutilate", "shoot", "slaughter", "stab", "torture", "violence"
            ],
            
            # Hate speech
            "hate_speech": [
                # Racial, homophobic, and other slurs represented by first letter
                # to avoid reproducing them while still allowing filtering
                "n-word", "f-slur", "c-slur", "r-slur", "k-slur", "g-slur",
                "tranny", "retard", "spic", "fag", "dyke", "homo", "retarded"
            ],
            
            # Drug references
            "drugs": [
                "cocaine", "crack", "heroin",
                "meth", "needle", "smack", "snort"
            ],
            
            # Image generation terms specifically for nude content
            "nude_generation": [
                "naked", "nude", "topless", "bottomless", "unclothed", "undressed"
                ,"undraped", "unclad", "birthday suit",
                "in the buff", "no clothes",
                "provocative", "risqué",
                "scantily", "suggestive", "without clothing", "without lingerie",
                "without bikini", "without swimsuit", "without bra", "without panties", "without underwear",
                "x-rated", "18+", "adult","lusty",
                "sultry", "arousing", "tantalizing","teasing",
                "undressing", "disrobing", "stripping", "strip", "striptease",
                "pinup", "playboy", "playmate", "centerfold"
            ],
            
            # Child safety terms - always block these in combination with generation requests
            "child_safety": [
                "child", "children", "kid", "kids", "toddler", "toddlers", "baby", 
                "babies", "infant", "infants", "minor", "minors", "teen", "teens", 
                "teenager", "teenagers", "adolescent", "adolescents", "young boy",
                "young girl", "little boy", "little girl", "youth", "underage",
                "pre-teen", "preteen", "tween", "middle schooler", "high schooler",
                "elementary", "kindergarten", "preschool", "daycare", "school girl",
                "school boy", "junior", "youngster"
            ]
        }
        
        # Common euphemisms
        self.euphemisms = [
            "adult content", "adult fun", "adult material",
            "bedroom activities", "between the sheets",
            "doing it", "get lucky", "getting busy",
            "happy ending", "horizontal dance", "intimate",
            "mature content", "netflix and chill", "no clothes",
            "not safe for work", "nsfw", "private parts", 
            "sleeping together", "special hug", "unclothed",
            "birthday suit", "in the buff", "eye candy",
            "showing skin", "wardrobe malfunction", "full frontal",
            "adult-oriented", "for mature audiences", "spicy content",
            "not suitable for minors", "artistic nude", "tasteful nude",
            "naturist", "boudoir",
            "bathing beauty", "adult entertainment",
            "adult model", "saucy", "come-hither",
            "dress-down", "see-through",
            "au naturel", "in one's skin", "undraped figure",
            "without anything on", "clothes optional", "tease", "uncensored", "disrobed"
        ]
        
        # Generation request markers
        self.generation_request_terms = [
            "generate", "create", "make", "produce", "render", 
            "show", "display", "visualize", "draw", "design",
            "depict", "illustrate", "portray", "show me", 
            "give me", "I want", "can you", "please make",
            "would like", "need a", "looking for"
        ]
        
        # Patterns for detecting attempts to bypass the filter
        self.regex_patterns = [
            r'\bn[\W_]*[i1l\|!][\W_]*[gq][\W_]*[gq9][\W_]*[e3a4][\W_]*r', # racial slur variations
            r'\bf[\W_]*[a4][\W_]*[gq][\W_]*[gq9][\W_]*[o0][\W_]*t',       # homophobic slur variations 
            r'\bc[\W_]*[u\*][\W_]*[n\*][\W_]*t',                          # gendered slur variations
            r'\bs[\W_]*[e3][\W_]*x',                                      # "sex" with characters between
            r'\bp[\W_]*[o0][\W_]*r[\W_]*n',                               # "porn" with characters between
            r'\bn[\W_]*[u\*][\W_]*d[\W_]*[e3]',                           # "nude" with characters between
            r'\bn[\W_]*[a4][\W_]*k[\W_]*[e3][\W_]*d',                     # "naked" with characters between
            r'\bs[\W_]*[e3][\W_]*x[\W_]*y',                               # "sexy" with characters between
            r'\bt[\W_]*[o0][\W_]*p[\W_]*l[\W_]*[e3][\W_]*s[\W_]*s',       # "topless" with characters between
            r'n[\W_]*s[\W_]*f[\W_]*w',                                    # "nsfw" with characters between
            r'b[\W_]*r[\W_]*[e3][\W_]*[a4][\W_]*s[\W_]*t',                # "breast" with characters between
            r'b[\W_]*[o0][\W_]*[o0][\W_]*b'                               # "boob" with characters between
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.regex_patterns]

        self.compiled_term_patterns = {}
        for category, terms in self.inappropriate_terms.items():
            self.compiled_term_patterns[category] = [
                re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) 
                for term in terms
            ]
        
    def is_appropriate(self, text, filter_level="moderate"):
        """
        Check if the provided text contains inappropriate content.
        
        Args:
            text (str): The text to check
            filter_level (str): 'strict', 'moderate', or 'minimal'
            
        Returns:
            bool: True if appropriate, False if inappropriate
        """
        if not text:
            return True
            
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Select categories based on filter level
        if filter_level == "strict":
            categories = ["profanity", "anatomical", "sexual_acts", "violence", "hate_speech", "drugs", "nude_generation", "child_safety"]
            check_euphemisms = True
            check_generation_requests = True
        elif filter_level == "moderate":
            categories = ["anatomical", "sexual_acts", "hate_speech", "drugs", "nude_generation", "child_safety"]
            check_euphemisms = True
            check_generation_requests = True
        elif filter_level == "minimal":
            categories = ["sexual_acts", "hate_speech", "nude_generation", "child_safety"]
            check_euphemisms = False
            check_generation_requests = True
        else:
            # Default to moderate
            categories = ["anatomical", "sexual_acts", "hate_speech", "drugs", "nude_generation", "child_safety"]
            check_euphemisms = True
            check_generation_requests = True
        
        # Check for exact matches in the selected categories
        for category in categories:
            if category in self.compiled_term_patterns:
                for pattern in self.compiled_term_patterns[category]:
                    if pattern.search(text_lower):
                        # If the category is nude_generation, check if there's also a generation request term
                        if category == "nude_generation" and check_generation_requests:
                            for gen_term in self.generation_request_terms:
                                if gen_term in text_lower:
                                    # Found both a nude term and a generation request term
                                    return False
                        else:
                            # For other categories, just finding the term is enough
                            return False
        
        # Check euphemisms if enabled
        if check_euphemisms:
            for euphemism in self.euphemisms:
                if euphemism in text_lower:
                    if check_generation_requests:
                        # Check if there's also a generation request term
                        for gen_term in self.generation_request_terms:
                            if gen_term in text_lower:
                                return False
                    else:
                        return False
        
        # Check regex patterns (for all filter levels)
        for pattern in self.compiled_patterns:
            if pattern.search(text_lower):
                return False
        
        # Special check for combinations that might indicate requests for inappropriate imagery
        if check_generation_requests:
            # Check for phrases like "woman without clothes" or "girl with no shirt"
            woman_terms = ["woman", "women", "girl", "girls", "female", "lady", "ladies", "model", "babe"]
            clothing_terms = ["clothes", "clothing", "dress", "shirt", "top", "pants", "skirt", "outfit"]
            negative_terms = ["without", "no", "not wearing", "lacking", "missing", "removed", "off"]
            
            for woman_term in woman_terms:
                if woman_term in text_lower:
                    for negative_term in negative_terms:
                        if negative_term in text_lower:
                            for clothing_term in clothing_terms:
                                if clothing_term in text_lower:
                                    for gen_term in self.generation_request_terms:
                                        if gen_term in text_lower:
                                            return False
            
            # Check for combinations like "generate attractive woman" with suggestive context
            attractive_terms = ["attractive", "beautiful", "pretty", "gorgeous", "hot", "sexy", "stunning"]
            suggestive_contexts = ["pose", "posing", "model", "beach", "shower", "bath", "bedroom", "bed", "swimming", 
                                  "bathing", "washing", "wet", "pool", "hot tub", "spa", "sauna", "sunbathing", 
                                  "changing", "dressing", "undressing", "locker room", "bathroom", "skinny dipping"]
            
            for woman_term in woman_terms:
                if woman_term in text_lower:
                    for attractive_term in attractive_terms:
                        if attractive_term in text_lower:
                            for context in suggestive_contexts:
                                if context in text_lower:
                                    for gen_term in self.generation_request_terms:
                                        if gen_term in text_lower:
                                            return False
            
            # CHILD SAFETY: Block ANY generation of children in potentially inappropriate contexts
            child_terms = self.inappropriate_terms["child_safety"]
            child_sensitive_contexts = ["bath", "bathing", "shower", "showering", "swimming", "pool", "beach", 
                                       "changing", "dressing", "undressing", "naked", "nude", "underwear", 
                                       "diaper", "clothes", "clothing", "bed", "bedroom", "sleeping", 
                                       "wet", "shirtless", "topless", "model", "posing", "pose", 
                                       "swimsuit", "bikini", "swim", "locker room", "bathroom"]
            
            # If ANY child term is present with ANY sensitive context AND a generation request, block it
            for child_term in child_terms:
                if child_term in text_lower:
                    for context in child_sensitive_contexts:
                        if context in text_lower:
                            for gen_term in self.generation_request_terms:
                                if gen_term in text_lower:
                                    return False
        
        return True
    

def add_watermark(image, text="Generated by Decompute Local", opacity=0.3, position="top-left"):
    """
    Add a watermark to an image.
    
    Args:
        image (PIL.Image): The image to watermark
        text (str): Watermark text
        opacity (float): Watermark opacity (0-1)
        position (str): Watermark position ('bottom-right', 'bottom-left', 'top-right', 'top-left', 'center')
        
    Returns:
        PIL.Image: The watermarked image
    """
    # Create a copy of the image to avoid modifying the original
    watermarked = image.copy()
    
    # Create a transparent layer for the watermark
    watermark_layer = Image.new('RGBA', watermarked.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)
    
    # Determine font size based on image dimensions - ENLARGED by multiplying by 1.5
    font_size = int(min(watermarked.size) / 25 * 3.5)
    
    try:
        # Try to use a system font (adjust path as needed)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Get text size
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else draw.textbbox((0, 0), text, font=font)[2:4]
    
    # Calculate position
    padding = 20  # Padding from the edges
    if position == "bottom-right":
        position = (watermarked.width - text_width - padding, watermarked.height - text_height - padding)
    elif position == "bottom-left":
        position = (padding, watermarked.height - text_height - padding)
    elif position == "top-right":
        position = (watermarked.width - text_width - padding, padding)
    elif position == "top-left":
        position = (padding, padding)
    else:  # center
        position = ((watermarked.width - text_width) // 2, (watermarked.height - text_height) // 2)
    
    # Draw text with shadow for better visibility
    # Shadow
    shadow_offset = max(1, int(font_size/15))
    draw.text((position[0] + shadow_offset, position[1] + shadow_offset), text, font=font, fill=(0, 0, 0, int(255 * opacity)))
    # Main text
    draw.text(position, text, font=font, fill=(255, 255, 255, int(255 * opacity)))
    
    # Composite the watermark layer onto the image
    if watermarked.mode != 'RGBA':
        watermarked = watermarked.convert('RGBA')
    
    watermarked = Image.alpha_composite(watermarked, watermark_layer)
    
    # Convert back to RGB if needed for saving in formats that don't support alpha
    if image.mode == 'RGB':
        watermarked = watermarked.convert('RGB')
    
    return watermarked



from PIL import Image
from PIL.ExifTags import TAGS
from piexif import dump, load
import json
import piexif
import io
import base64
from datetime import datetime

import io
import json
from datetime import datetime
from typing import Dict, Union

from PIL import Image, PngImagePlugin
import piexif

# ------------------------------------------------------------------#
# 1.  WRITING METADATA                                              #
# ------------------------------------------------------------------#


def _build_png_info(metadata: Dict, xmp_str: str = "") -> PngImagePlugin.PngInfo:
    """
    Return a PngInfo object that carries the JSON metadata (tEXt chunk)
    plus a few human‑readable fields.
    """
    info = PngImagePlugin.PngInfo()

    # mandatory human‑readable fields
    now = datetime.now().strftime("%Y‑%m‑%d %H:%M:%S")
    info.add_text("Software", "Decompute Local Image Generator")
    info.add_text("Creation Time", now)

    # single JSON blob – easy round‑trip
    info.add_text("Decompute_Metadata", json.dumps(metadata))

    if xmp_str:
        info.add_text("XML:com.adobe.xmp", xmp_str)

    return info


def add_metadata_to_image(
    img: Image.Image, metadata: Dict, *, keep_format: str = "PNG"
) -> Image.Image:
    """
    Return a **new** Pillow image with user metadata embedded.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image (any mode/size).
    metadata : dict
        Anything JSON‑serialisable – prompt, model, params, etc.
    keep_format : str
        "PNG" (default) or "JPEG" – forces the output file format.
    """
    now_iso = datetime.now().strftime("%Y‑%m‑%dT%H:%M:%S")
    img_copy = img.copy()

    # ------------- PNG PATH ---------------------------------------#
    if keep_format.upper() == "PNG":
        png_info = _build_png_info(metadata)
        with io.BytesIO() as buf:
            img_copy.save(buf, format="PNG", pnginfo=png_info)
            buf.seek(0)
            tagged = Image.open(buf)
            tagged.load()
        return tagged

    # ------------- JPEG PATH --------------------------------------#
    keep_format = "JPEG"  # force correct spelling
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Put app / time info in standard 0th tags
    exif_dict["0th"][piexif.ImageIFD.Software] = "Decompute Local Image Generator 2"
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = (
        f"Decompute_Metadata {now_iso}"
    )

    # JSON payload in UserComment (Exif)
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = json.dumps(metadata).encode("utf‑8")
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = now_iso.replace("T", " ")

    exif_bytes = piexif.dump(exif_dict)

    with io.BytesIO() as buf:
        img_copy.save(
            buf,
            format="JPEG",
            exif=exif_bytes,
            quality=95,
            optimize=True,
            progressive=True,
            subsampling=0,
        )
        buf.seek(0)
        tagged = Image.open(buf)
        tagged.load()
    return tagged


# ------------------------------------------------------------------#
# 2.  READING  METADATA                                             #
# ------------------------------------------------------------------#
import os

def extract_metadata_from_image(src: Union[str, os.PathLike, Image.Image]) -> Dict:
    """
    Read back the metadata written by `add_metadata_to_image`.

    Returns an **empty dict** if no payload is found.
    """
    if isinstance(src, (str, os.PathLike, bytes)):
        img = Image.open(str(src))      # PathLike → str
    else:
        img = src

    meta: Dict = {}

    # ------------ PNG  (tEXt/iTXt) --------------------------------#
    # ------------ PNG (tEXt/iTXt) --------------------------------#
    if img.format == "PNG":
        # Pillow ≥10 stores text chunks in .text; <10 in .info
        # Use whichever actually contains data
        if hasattr(img, "text") and img.text:
            text_chunks = img.text          # Pillow 10+
        else:
            text_chunks = img.info          # Pillow 9.x and earlier

        payload = text_chunks.get("Decompute_Metadata")
        if payload:
            try:
                meta = json.loads(payload)
            except json.JSONDecodeError:
                meta["raw_payload"] = payload
        return meta


    # ------------ JPEG  (Exif.UserComment) ------------------------#
    exif_bytes = img.info.get("exif")
    if exif_bytes:
        try:
            exif_dict = piexif.load(exif_bytes)
            user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
            if user_comment:
                if isinstance(user_comment, bytes):
                    user_comment = user_comment.decode("utf‑8", errors="ignore")
                try:
                    meta = json.loads(user_comment)
                except json.JSONDecodeError:
                    meta["raw_payload"] = user_comment
        except Exception:
            pass
    return meta

def _dict_to_pnginfo(chunks: Dict[str, str]) -> PngImagePlugin.PngInfo:
    pi = PngImagePlugin.PngInfo()
    for k, v in chunks.items():
        pi.add_text(k, v)
    return pi


def save_with_metadata(
    img: Image.Image,
    metadata: Dict,
    outfile: Union[str, os.PathLike],
    *,
    fmt: str = "PNG",
) -> None:
    """Embed metadata and write the file in one step."""
    tagged = add_metadata_to_image(img, metadata, keep_format=fmt)

    if fmt.upper() == "PNG":
        # Pillow ≥10 stores chunks in .text; <10 in .info
        chunks_dict = getattr(tagged, "text", None) or tagged.info
        pnginfo_obj = _dict_to_pnginfo(chunks_dict)
        tagged.save(outfile, pnginfo=pnginfo_obj)
    else:  # JPEG keeps EXIF automatically
        tagged.save(outfile)


def dict_to_pnginfo(chunks: dict) -> PngImagePlugin.PngInfo:
    """Build a PngInfo object from a plain text‑chunk dict."""
    pi = PngImagePlugin.PngInfo()
    for k, v in chunks.items():
        pi.add_text(k, v)
    return pi


def save_png_with_chunks(img: Image.Image, fp, **save_kwargs) -> None:
    """
    Save *img* (PNG) to *fp* while preserving custom text chunks.
    *fp* can be a filename/path or a BytesIO buffer.
    """
    chunks = getattr(img, "text", None) or img.info
    pnginfo = dict_to_pnginfo(chunks)
    img.save(fp, format="PNG", pnginfo=pnginfo, **save_kwargs)
