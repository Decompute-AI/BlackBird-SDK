import os
import json
import requests
from typing import Dict, List, Optional
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import logging
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import warnings
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionLLMProcessor:
    """
    A dedicated processor for Vision LLM inference using SmolVLM model.
    """
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the SmolVLM model once with optimized GPU configuration"""
        try:
            # Enhanced GPU detection and configuration
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"🚀 CUDA Available! Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"🔧 CUDA Version: {torch.version.cuda}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                
                # Clear GPU cache for optimal performance
                torch.cuda.empty_cache()
                
                # Set memory fraction to avoid OOM errors
                torch.cuda.set_per_process_memory_fraction(0.8)
                
            else:
                self.device = "cpu"
                print("⚠️  CUDA not available, using CPU")
            
            print(f"📥 Loading SmolVLM model on {self.device}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
            
            # Enhanced model loading with GPU optimizations
            if self.device == "cuda":
                # Try Flash Attention 2 first, fallback to standard attention
                try:
                    print("🔥 Attempting to load with Flash Attention 2...")
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        "HuggingFaceTB/SmolVLM-500M-Instruct",
                        torch_dtype=torch.float16,  # Use float16 for faster GPU inference
                        device_map="auto",  # Automatically distribute model across available GPUs
                        _attn_implementation="flash_attention_2",  # Use Flash Attention 2 for speed
                        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                    )
                    print("✅ Flash Attention 2 loaded successfully")
                except Exception as flash_error:
                    print(f"⚠️  Flash Attention 2 not available: {flash_error}")
                    print("🔄 Falling back to standard attention...")
                    
                    # Fallback to standard attention
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        "HuggingFaceTB/SmolVLM-500M-Instruct",
                        torch_dtype=torch.float16,  # Use float16 for faster GPU inference
                        device_map="auto",  # Automatically distribute model across available GPUs
                        _attn_implementation="eager",  # Use standard attention as fallback
                        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                    )
                    print("✅ Standard attention loaded successfully")
                
                # Enable optimizations
                self.model.eval()  # Set to evaluation mode
                
                # Compile model for faster inference (PyTorch 2.0+)
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("✅ Model compiled for faster inference")
                except Exception as compile_error:
                    print(f"⚠️  Model compilation failed (continuing without): {compile_error}")
                
            else:
                # CPU configuration
                self.model = AutoModelForVision2Seq.from_pretrained(
                    "HuggingFaceTB/SmolVLM-500M-Instruct",
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="eager",
                ).to(self.device)
            
            print(f"✅ SmolVLM model loaded successfully on {self.device}")
            
            # Print memory usage if on GPU
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"📊 GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
        except Exception as e:
            print(f"❌ Failed to initialize SmolVLM model: {e}")
            logger.error(f"SmolVLM initialization failed: {e}")
            # Set to None to indicate failure
            self.processor = None
            self.model = None
            self.device = None

    def get_vision_description(self, image_path: str) -> str:
        """
        Get image description using SmolVLM model with GPU optimization.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Description of the image from Vision LLM
        """
        try:
            if self.model is None or self.processor is None:
                return "Vision LLM model not available - using fallback description"
            
            print(f"\n=== 🚀 GPU Vision LLM Processing Started for: {image_path} ===")
            
            # Load and process the image
            image = Image.open(image_path)
            
            # Create input messages using chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the content of this image in detail. Focus on text, objects, layout, and any visual elements present."}
                    ]
                }
            ]
            
            # Prepare inputs with GPU optimization
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            
            # Move inputs to device (GPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Enhanced generation parameters for GPU
            generation_config = {
                "max_new_tokens": 500,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,  # Enable KV cache for faster generation
            }
            
            # Add GPU-specific optimizations
            if self.device == "cuda":
                generation_config.update({
                    "num_beams": 1,  # Use greedy decoding for speed on GPU
                    "early_stopping": True,
                })
            
            # Generate description with optimized settings
            print(f"🔄 Generating description on {self.device}...")
            start_inference = time.time()
            
            with torch.no_grad():
                # Use autocast for mixed precision on GPU
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        generated_ids = self.model.generate(
                            **inputs,
                            **generation_config
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        **generation_config
                    )
            
            inference_time = time.time() - start_inference
            
            # Decode the response
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
            # Extract just the assistant's response
            response = generated_texts[0]
            
            # Clean up the response to remove the prompt part
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            elif "assistant" in response.lower():
                parts = response.lower().split("assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()
            
            # Print Vision LLM results with performance metrics
            print(f"✅ Vision LLM DONE for: {image_path}")
            print(f"⚡ Inference time: {inference_time:.2f} seconds")
            if self.device == "cuda":
                memory_used = torch.cuda.max_memory_allocated(0) / 1024**3
                print(f"📊 Peak GPU memory: {memory_used:.2f} GB")
            
            print(f"🤖 Vision Description:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Clear GPU cache after inference
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            error_msg = f"Vision LLM Error: {str(e)}"
            print(f"❌ Vision LLM FAILED for: {image_path}")
            print(f"Error: {error_msg}")
            logger.error(f"Vision LLM request failed for {image_path}: {e}")
            
            # Clear GPU cache on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return error_msg

class info_preprocess:
    """
    A class for preprocessing various types of artifacts (images, PDFs, documents)
    using OCR and Vision LLM for information extraction.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None, vision_llm_url: str = "http://localhost:8000/vision"):
        """
        Initialize the info_preprocess class.
        
        Args:
            tesseract_path: Path to tesseract executable (optional)
            vision_llm_url: URL for the vision LLM API endpoint
        """
        self.vision_llm_url = vision_llm_url
        
        # Initialize Vision LLM processor
        try:
            self.vision_processor = VisionLLMProcessor()
            logger.info("Vision LLM processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vision LLM processor: {e}")
            self.vision_processor = None
        
        # Configure Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Verify Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not found or not properly configured: {e}")
            raise
    
    def get_image_description(self, image_path: str) -> str:
        """
        Extract text description from an image using OCR and Vision LLM.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Text description of the image content
        """
        try:
            # Validate file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Extract text using OCR
            ocr_text = self._extract_text_with_ocr(image_path)
            
            # Get description using Vision LLM
            vision_description = self._get_vision_llm_description(image_path)
            
            # Combine results
            combined_description = f"OCR Text: {ocr_text}\n\nVision Description: {vision_description}"
            
            # Print final summary
            print(f"\n🎯 FINAL RESULT for: {image_path}")
            print("=" * 60)
            print("Combined OCR + Vision LLM Analysis:")
            print("-" * 40)
            print(combined_description)
            print("=" * 60)
            
            logger.info(f"Successfully processed image: {image_path}")
            return combined_description
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def get_pdf_info(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract information from PDF pages using OCR and Vision LLM.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict[int, str]: Dictionary with page numbers as keys and descriptions as values
        """
        try:
            # Validate file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            page_descriptions = {}
            
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            
            # Get total number of pages
            total_pages = len(pdf_document)
            logger.info(f"Processing PDF with {total_pages} pages: {pdf_path}")
            
            # Safety check: limit to reasonable number of pages to prevent infinite loops
            max_pages = 100  # Adjust this limit as needed
            if total_pages > max_pages:
                logger.warning(f"PDF has {total_pages} pages, limiting processing to first {max_pages} pages")
                total_pages = max_pages
            
            for page_num in range(total_pages):
                try:
                    page = pdf_document[page_num]
                    
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                    img_data = pix.tobytes("png")
                    
                    # Save temporary image
                    temp_image_path = f"temp_page_{page_num}.png"
                    with open(temp_image_path, "wb") as f:
                        f.write(img_data)
                    
                    try:
                        # Extract text using OCR
                        ocr_text = self._extract_text_with_ocr(temp_image_path)
                        
                        # Get description using Vision LLM
                        vision_description = self._get_vision_llm_description(temp_image_path)
                        
                        # Combine results
                        page_description = f"OCR Text: {ocr_text}\n\nVision Description: {vision_description}"
                        page_descriptions[page_num + 1] = page_description
                        
                        # Print page summary
                        print(f"\n📄 PAGE {page_num + 1} COMPLETE")
                        print(f"📊 OCR Text Length: {len(ocr_text)} characters")
                        print(f"🤖 Vision Description Length: {len(vision_description)} characters")
                        
                        logger.info(f"Processed page {page_num + 1}/{total_pages}")
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    page_descriptions[page_num + 1] = f"Error processing page: {str(e)}"
                    continue
            
            pdf_document.close()
            
            logger.info(f"Successfully processed PDF: {pdf_path} ({len(page_descriptions)} pages)")
            return page_descriptions
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def get_doc_info(self, doc_path: str) -> Dict[int, str]:
        """
        Extract information from document pages using OCR and Vision LLM.
        This method handles various document formats by converting them to images.
        
        Args:
            doc_path: Path to the document file (PDF, DOC, DOCX, etc.)
            
        Returns:
            Dict[int, str]: Dictionary with page numbers as keys and descriptions as values
        """
        try:
            # Validate file exists
            if not os.path.exists(doc_path):
                raise FileNotFoundError(f"Document file not found: {doc_path}")
            
            # For now, we'll treat documents the same as PDFs
            # In a production environment, you might want to add support for other formats
            # like DOC, DOCX using libraries like python-docx or win32com
            
            return self.get_pdf_info(doc_path)
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            raise
    
    def _extract_text_with_ocr(self, image_path: str) -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        try:
            print(f"\n=== OCR Processing Started for: {image_path} ===")
            
            # Open image
            image = Image.open(image_path)
            
            # Configure OCR parameters for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:,.<>?/ '
            
            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Clean up text
            text = text.strip()
            
            if not text:
                text = "No text detected in image"
            
            # Print OCR results
            print(f"✓ OCR DONE for: {image_path}")
            print(f"📄 Extracted Text:")
            print("-" * 50)
            print(text)
            print("-" * 50)
            print(f"📊 Text Length: {len(text)} characters")
            
            return text
            
        except Exception as e:
            error_msg = f"OCR Error: {str(e)}"
            print(f"❌ OCR FAILED for: {image_path}")
            print(f"Error: {error_msg}")
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return error_msg
    
    def _get_vision_llm_description(self, image_path: str) -> str:
        """
        Get image description using Vision LLM.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Description of the image from Vision LLM
        """
        try:
            if self.vision_processor is not None:
                # Use the actual SmolVLM model
                return self.vision_processor.get_vision_description(image_path)
            else:
                # Fallback to dummy implementation
                return self._dummy_vision_llm_request(image_path)
                
        except Exception as e:
            error_msg = f"Vision LLM Error: {str(e)}"
            print(f"❌ Vision LLM FAILED for: {image_path}")
            print(f"Error: {error_msg}")
            logger.error(f"Vision LLM request failed for {image_path}: {e}")
            return error_msg
    
    def _dummy_vision_llm_request(self, image_path: str) -> str:
        """
        Dummy method for Vision LLM request and response.
        Used as fallback when the actual model is not available.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Dummy description of the image
        """
        # This is a dummy implementation
        # Used as fallback when SmolVLM model is not available
        
        # Simulate API request
        logger.info(f"Making dummy Vision LLM request for: {image_path}")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Return dummy description based on file name
        filename = os.path.basename(image_path)
        
        # Check if this is a temporary page file from PDF processing
        if filename.startswith("temp_page_") and filename.endswith(".png"):
            # Extract page number from filename (temp_page_0.png -> page 1)
            try:
                page_num = int(filename.replace("temp_page_", "").replace(".png", "")) + 1
                return f"Dummy Vision LLM description for PDF page {page_num}: This appears to be a document page containing text, images, and visual elements."
            except ValueError:
                return f"Dummy Vision LLM description for document page: {filename}"
        elif "page" in filename.lower():
            return f"Dummy Vision LLM description for document page: {filename}"
        else:
            return f"Dummy Vision LLM description for image: {filename}. This appears to be a document or image containing text and visual elements."
    
    def process_batch(self, file_paths: List[str]) -> Dict[str, Dict]:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dict[str, Dict]: Results for each file
        """
        results = {}
        
        for file_path in file_paths:
            try:
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    results[file_path] = {
                        'type': 'image',
                        'description': self.get_image_description(file_path)
                    }
                elif file_ext == '.pdf':
                    results[file_path] = {
                        'type': 'pdf',
                        'pages': self.get_pdf_info(file_path)
                    }
                else:
                    results[file_path] = {
                        'type': 'document',
                        'pages': self.get_doc_info(file_path)
                    }
                    
            except Exception as e:
                results[file_path] = {
                    'error': str(e)
                }
        
        return results


def test_tesseract_ocr():
    """
    Independent test function for Tesseract OCR functionality.
    """
    print("=== Testing Tesseract OCR ===")
    
    # Test OCR on a sample image (you'll need to provide an actual image)
    test_image_path = "test_image.png"  # Replace with actual test image
    
    if os.path.exists(test_image_path):
        try:
            # Initialize the class
            processor = info_preprocess()
            
            # Test OCR extraction
            ocr_text = processor._extract_text_with_ocr(test_image_path)
            print(f"OCR Result: {ocr_text}")
            
        except Exception as e:
            print(f"OCR Test failed: {e}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please provide a test image to run OCR test")


def test_vision_llm():
    """
    Independent test function for Vision LLM functionality.
    """
    print("=== Testing Vision LLM ===")
    
    # Test Vision LLM on a sample image
    test_image_path = "test_image.png"  # Replace with actual test image
    
    if os.path.exists(test_image_path):
        try:
            # Initialize the class
            processor = info_preprocess()
            
            # Test Vision LLM description
            vision_description = processor._get_vision_llm_description(test_image_path)
            print(f"Vision LLM Result: {vision_description}")
            
        except Exception as e:
            print(f"Vision LLM Test failed: {e}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please provide a test image to run Vision LLM test")


def test_pdf_processing():
    """
    Independent test function for PDF processing.
    """
    print("=== Testing PDF Processing ===")
    
    # Test PDF processing
    test_pdf_path = "test_document.pdf"  # Replace with actual test PDF
    
    if os.path.exists(test_pdf_path):
        try:
            # Initialize the class
            processor = info_preprocess()
            
            # Test PDF processing
            pdf_info = processor.get_pdf_info(test_pdf_path)
            print(f"PDF Processing Result:")
            for page_num, description in pdf_info.items():
                print(f"Page {page_num}: {description[:100]}...")
            
        except Exception as e:
            print(f"PDF Processing Test failed: {e}")
    else:
        print(f"Test PDF not found: {test_pdf_path}")
        print("Please provide a test PDF to run PDF processing test")


def main():
    """
    Main function to demonstrate usage and run tests.
    """
    print("=== Artifact Processing Demo ===")
    
    # Example usage
    processor = info_preprocess()
    
    # Example file paths (replace with actual files)
    test_files = [
        "C:/Users/Administrator/Pictures/Screenshots/test_image.png",
        "C:/Users/Administrator/Downloads/Attention_paper.pdf"
        # ,"sample_doc.docx"

    ]
    
    print("Processing files...")
    results = processor.process_batch(test_files)
    
    # Print results
    for file_path, result in results.items():
        print(f"\nFile: {file_path}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Type: {result['type']}")
            if result['type'] == 'image':
                print(f"Description: {result['description'][:200]}...")
            else:
                print(f"Pages: {len(result['pages'])}")
                for page_num, description in list(result['pages'].items())[:2]:  # Show first 2 pages
                    print(f"  Page {page_num}: {description[:100]}...")


# if __name__ == "__main__":
#     # Run tests
#     test_tesseract_ocr()
#     test_vision_llm()
#     test_pdf_processing()
    
#     # Run main demo
#     main()
