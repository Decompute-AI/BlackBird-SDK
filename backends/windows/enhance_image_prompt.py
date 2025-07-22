import re
import json
import random
from typing import Dict, List, Optional, Union, Tuple

class AdvancedPromptEnhancer:
    """
    Advanced prompt enhancement with more sophisticated techniques including:
    - Structured prompt formatting
    - Subject/style/modifier separation
    - Template-based enhancements
    - Random variation for creative results
    """
    
    def __init__(self, config_file="prompt_config.json"):
        """
        Initialize with optional configuration file
        """
        self.config = self._load_config(config_file)
        self.templates = self.config.get("templates", {})
        self.modifiers = self.config.get("modifiers", {})
        self.weights = self.config.get("weights", {})
        
    def _load_config(self, config_file="prompt_config.json"):
        """Load configuration from file or use defaults"""
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
                
        # Default configuration
        return {
            "templates": {
                "default": "{subject}, {style}, {quality}",
                "portrait": "portrait of {subject}, {style}, {quality}, {lighting}",
                "landscape": "landscape of {subject}, {style}, {quality}, {atmosphere}",
                "product": "product photo of {subject}, {style}, {quality}, {lighting}, {background}"
            },
            "modifiers": {
                "quality": [
                    "highly detailed", "photorealistic", "8K resolution", 
                    "ultra HD", "masterpiece", "professional quality"
                ],
                "style": {
                    "photo": ["DSLR photo", "professional photography", "photorealistic"],
                    "painting": ["oil painting", "digital art", "concept art", "matte painting"],
                    "3d": ["3D render", "octane render", "unreal engine", "ray tracing"],
                    "anime": ["anime style", "manga style", "Japanese animation"],
                    "cartoon": ["cartoon style", "cel shaded", "animated", "Pixar style"]
                },
                "lighting": [
                    "dramatic lighting", "cinematic lighting", "soft lighting", 
                    "golden hour", "studio lighting", "natural lighting"
                ],
                "atmosphere": [
                    "atmospheric", "moody", "foggy", "clear sky", "sunrise", "sunset"
                ],
                "background": [
                    "studio background", "gradient background", "solid color background",
                    "blurred background", "minimalist background"
                ]
            },
            "weights": {
                "subject": 1.0,
                "quality": 0.7,
                "style": 0.8,
                "lighting": 0.6,
                "atmosphere": 0.5,
                "background": 0.4
            }
        }
    
    def _extract_components(self, prompt: str) -> Dict[str, str]:
        """Extract semantic components from the prompt"""
        # Simple heuristic approach - could be enhanced with NLP
        components = {"subject": prompt}
        
        # Check for templates
        for template_name, _ in self.templates.items():
            if template_name.lower() in prompt.lower():
                components["template"] = template_name
                break
                
        # Check for styles
        for style_name, style_keywords in self.modifiers.get("style", {}).items():
            for keyword in style_keywords:
                if keyword.lower() in prompt.lower():
                    components["style"] = style_name
                    break
            if "style" in components:
                break
                
        return components
    
    def _select_modifiers(self, category: str, count: int, existing_prompt: str) -> List[str]:
        """Select modifiers that aren't already in the prompt"""
        if category not in self.modifiers:
            return []
            
        category_modifiers = self.modifiers[category]
        
        # Handle nested dictionaries for style modifiers
        if isinstance(category_modifiers, dict):
            # Default to photo style if no specific style
            style_name = "photo"
            category_modifiers = category_modifiers.get(style_name, [])
            
        # Filter out modifiers already in the prompt
        filtered_modifiers = [
            m for m in category_modifiers 
            if m.lower() not in existing_prompt.lower()
        ]
        
        # Select random subset
        if len(filtered_modifiers) <= count:
            return filtered_modifiers
        return random.sample(filtered_modifiers, count)
    
    def _apply_template(self, components: Dict[str, str], template_name: str = "default") -> str:
        """Apply a template to the components"""
        template = self.templates.get(template_name, self.templates["default"])
        
        # Get the subject
        subject = components.get("subject", "")
        
        # Select style modifiers
        style_name = components.get("style", "photo")
        style_modifiers = self._select_modifiers(f"style", 1, subject)
        if style_modifiers:
            style = style_modifiers[0]
        else:
            # Default style based on style_name
            style = f"{style_name} style"
            
        # Select quality modifiers
        quality_modifiers = self._select_modifiers("quality", 2, subject)
        quality = ", ".join(quality_modifiers)
        
        # Prepare other modifiers
        lighting = ", ".join(self._select_modifiers("lighting", 1, subject))
        atmosphere = ", ".join(self._select_modifiers("atmosphere", 1, subject))
        background = ", ".join(self._select_modifiers("background", 1, subject))
        
        # Format the template
        result = template.format(
            subject=subject,
            style=style,
            quality=quality,
            lighting=lighting,
            atmosphere=atmosphere,
            background=background
        )
        
        # Clean up any empty placeholders
        result = re.sub(r', ,', ',', result)
        result = re.sub(r', $', '', result)
        result = re.sub(r' ,', ',', result)
        
        return result
        
    def _apply_weights(self, components: Dict[str, Tuple[str, float]]) -> str:
        """Apply weights to components"""
        weighted_parts = []
        
        for component_type, (text, weight) in components.items():
            if not text:
                continue
                
            # Apply default weight from config if available
            if weight is None and component_type in self.weights:
                weight = self.weights[component_type]
                
            # Apply the weight if specified
            if weight is not None and weight != 1.0:
                weighted_parts.append(f"({text}:{weight:.1f})")
            else:
                weighted_parts.append(text)
                
        return ", ".join(weighted_parts)
        
    def enhance(self, prompt: str, template: str = None, 
                style: str = None, apply_weights: bool = True) -> str:
        """
        Enhance a prompt using advanced techniques
        
        Args:
            prompt: Original user prompt
            template: Template to use (default, portrait, landscape, product)
            style: Style override (photo, painting, 3d, anime, cartoon)
            apply_weights: Whether to apply weights to components
            
        Returns:
            Enhanced prompt
        """
        # Extract components
        components = self._extract_components(prompt)
        
        # Override with provided parameters
        if template:
            components["template"] = template
        if style:
            components["style"] = style
            
        # Apply template
        template_name = components.get("template", "default")
        enhanced_prompt = self._apply_template(components, template_name)
        
        # Apply weights if requested
        if apply_weights:
            # Create weighted components
            weighted_components = {
                "subject": (prompt, 1.0),
                "style": (components.get("style", ""), 0.8),
                "quality": (", ".join(self._select_modifiers("quality", 2, prompt)), 0.7),
                "lighting": (", ".join(self._select_modifiers("lighting", 1, prompt)), 0.6)
            }
            
            enhanced_prompt = self._apply_weights(weighted_components)
            
        return enhanced_prompt


def load_enhancer_config(config_path):
    """
    Load the enhancer configuration from a JSON file
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading enhancer config: {e}")
        return None


def get_enhancer(config_path=None):
    """
    Factory function to get the appropriate enhancer based on configuration
    """
    return AdvancedPromptEnhancer(config_path)


# Example usage
# if __name__ == "__main__":
#     enhancer = AdvancedPromptEnhancer(config_file="app-demo/prompt_config.json")
    
#     # Example enhancements
#     examples = [
#         "a cat sitting on a windowsill",
#         "sunset over mountains",
#         "portrait of a woman with red hair"
#     ]
    
#     for example in examples:
#         enhanced = enhancer.enhance(example)
#         print(f"Original: {example}")
#         print(f"Enhanced: {enhanced}")
#         print()
