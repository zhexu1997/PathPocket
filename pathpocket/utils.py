"""
Utility functions for PathPocket
File I/O, content processing, and helper functions
Copied from raganything.utils and adapted for PathPocket
"""

import json
import tiktoken
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple

from pathpocket.core import logger


class Tokenizer:
    """Base tokenizer interface"""
    def __init__(self, model_name: str = "gpt-4o-mini", tokenizer=None):
        self.model_name = model_name
        self.tokenizer = tokenizer
    
    def encode(self, text: str) -> list:
        """Encode text to tokens"""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return []
    
    def decode(self, tokens: list) -> str:
        """Decode tokens to text"""
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        return ""


class TiktokenTokenizer(Tokenizer):
    """Tiktoken-based tokenizer implementation"""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
            super().__init__(model_name=model_name, tokenizer=tokenizer)
        except KeyError:
            # Fallback to cl100k_base encoding
            tokenizer = tiktoken.get_encoding("cl100k_base")
            super().__init__(model_name=model_name, tokenizer=tokenizer)


def read_content_list_from_json(json_file: Path, method_dir: Path = None):
    """
    Read content_list from JSON file and fix image paths

    Args:
        json_file: JSON file path
        method_dir: Method directory (e.g., auto/), for determining image base path

    Returns:
        content_list: Content list
    """
    if method_dir is None:
        method_dir = json_file.parent

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            content_list = json.load(f)

        # Fix relative paths to absolute paths
        for item in content_list:
            if isinstance(item, dict):
                for field_name in [
                    "img_path",
                    "table_img_path",
                ]:
                    if field_name in item and item[field_name]:
                        img_path = item[field_name]

                        # If absolute path, use directly
                        if Path(img_path).is_absolute():
                            if not Path(img_path).exists():
                                img_name = Path(img_path).name
                                potential_path = method_dir / "images" / img_name
                                if not potential_path.exists():
                                    potential_path = method_dir / img_name
                                if potential_path.exists():
                                    item[field_name] = str(potential_path.resolve())
                            continue

                        # Handle relative paths
                        if img_path.startswith("images/"):
                            absolute_img_path = (method_dir / img_path).resolve()
                        else:
                            potential_path1 = (method_dir / "images" / img_path).resolve()
                            potential_path2 = (method_dir / img_path).resolve()

                            if potential_path1.exists():
                                absolute_img_path = potential_path1
                            elif potential_path2.exists():
                                absolute_img_path = potential_path2
                            else:
                                absolute_img_path = (method_dir / "images" / img_path).resolve()

                        item[field_name] = str(absolute_img_path)

                        if not Path(absolute_img_path).exists():
                            print(f"Warning: Image file does not exist: {absolute_img_path}")

        return content_list
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_titles_with_content(content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge titles with all subsequent content until the next title
    Preserves paragraph boundaries using double newlines
    """
    merged_list = []
    i = 0

    while i < len(content_list):
        item = content_list[i]

        if item.get("type") != "text":
            merged_list.append(item)
            i += 1
            continue

        text = item.get("text", "").strip()
        text_level = item.get("text_level", 0)

        if text_level > 0 and text:
            merged_text_parts = [text]
            j = i + 1
            has_content = False

            while j < len(content_list):
                next_item = content_list[j]

                if next_item.get("type") != "text":
                    # Stop at non-text items to preserve structure
                    break

                next_text = next_item.get("text", "").strip()
                next_text_level = next_item.get("text_level", 0)

                if next_text_level > 0:
                    # Stop at next title
                    break

                if next_text:
                    merged_text_parts.append(next_text)
                    has_content = True
                    j += 1
                else:
                    j += 1
                    continue

            if has_content:
                # Use double newline to preserve paragraph boundaries
                merged_text = "\n".join(merged_text_parts)
                merged_item = item.copy()
                merged_item["text"] = merged_text
                merged_item["_merged"] = True
                merged_list.append(merged_item)
                i = j
            else:
                i += 1
        else:
            merged_list.append(item)
            i += 1

    return merged_list


def merge_small_chunks(
    content_list: List[Dict[str, Any]], 
    max_tokens: int = 4800,
    tokenizer=None,
    tiktoken_model_name: str = "gpt-4o-mini"
) -> List[Dict[str, Any]]:
    """
    Merge chunks with fewer tokens than the specified threshold
    Processes all text items first, then appends non-text items at the end in their original order
    
    Args:
        content_list: List of content items
        max_tokens: Maximum tokens threshold. Only chunks smaller than this will be merged.
                   Chunks at or above this threshold are kept intact to preserve paragraph integrity.
        tokenizer: Tokenizer instance. If None, will create a TiktokenTokenizer.
        tiktoken_model_name: Model name for tokenizer if tokenizer is None (default: "gpt-4o-mini")
    
    Returns:
        Processed content list with small chunks merged, non-text items at the end
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        # Create TiktokenTokenizer
        tokenizer = TiktokenTokenizer(model_name=tiktoken_model_name)
    
    # Separate text and non-text items
    text_items = []
    non_text_items = []
    
    for item in content_list:
        if item.get("type") == "text" and item.get("text", "").strip():
            text_items.append(item)
        else:
            # Non-text items or empty text items
            non_text_items.append(item)
    
    # Process text items: merge small chunks
    merged_list = []
    i = 0

    while i < len(text_items):
        item = text_items[i]
        text = item.get("text", "").strip()
        
        if not text:
            i += 1
            continue

        # Calculate token count for current text
        try:
            text_tokens = tokenizer.encode(text)
            text_token_count = len(text_tokens)
        except Exception as e:
            # Fallback to character count if tokenization fails
            print(f"Warning: Tokenization failed, using character count: {e}")
            text_token_count = len(text) // 4  # Rough estimate: 1 token ≈ 4 characters

        # Only merge if current chunk is smaller than threshold
        # Chunks at or above threshold are kept intact to preserve paragraph integrity
        if text_token_count < max_tokens:
            merged_text_parts = [text]
            merged_token_count = text_token_count
            merged_item = item.copy()
            j = i + 1

            # Continue merging subsequent text chunks
            # Skip non-text items (they will be added at the end)
            while j < len(text_items):
                next_item = text_items[j]
                next_text = next_item.get("text", "").strip()
                
                if not next_text:
                    j += 1
                    continue

                # Calculate token count for next text
                try:
                    next_tokens = tokenizer.encode(next_text)
                    next_token_count = len(next_tokens)
                except Exception as e:
                    # Fallback to character count if tokenization fails
                    next_token_count = len(next_text) // 4

                # Use newline to preserve paragraph boundaries
                potential_merged = "\n".join(merged_text_parts + [next_text])
                
                # Calculate total token count for merged text
                try:
                    potential_tokens = tokenizer.encode(potential_merged)
                    potential_token_count = len(potential_tokens)
                except Exception as e:
                    # Fallback: use sum of individual token counts + separator overhead
                    potential_token_count = merged_token_count + next_token_count + 2  # +2 for separator
                
                # Check if adding this text would exceed threshold
                if potential_token_count > max_tokens:
                    # Would exceed threshold, stop merging to preserve paragraph integrity
                    break
                else:
                    # Can add this text, continue merging
                    merged_text_parts.append(next_text)
                    merged_token_count = potential_token_count
                    j += 1
                    # If we've reached or exceeded threshold, stop to preserve paragraph integrity
                    if potential_token_count >= max_tokens:
                        break

            # Merge text parts with newline separator
            merged_text = "\n".join(merged_text_parts)
            merged_item["text"] = merged_text
            merged_item["_merged_small"] = True
            merged_list.append(merged_item)

            i = j
        else:
            # Text chunk is already at or above threshold
            # Keep it as-is to preserve paragraph integrity (don't break it up)
            merged_list.append(item)
            i += 1

    # Append all non-text items at the end in their original order
    merged_list.extend(non_text_items)

    return merged_list


def find_all_parsed_files(output_dir: Path, parse_methods: list = ["auto", "ocr", "txt", "vlm"], prefer_enhanced: bool = True, enhanced_suffix: str = "_enhanced"):
    """
    Find all parsed output files
    If prefer_enhanced is True, will prefer enhanced directory (e.g., WHO_enhanced) over original directory
    
    Args:
        output_dir: Base output directory (e.g., /path/to/Guidelines/WHO)
        parse_methods: List of parse methods to look for
        prefer_enhanced: If True, prefer enhanced directory over original
        enhanced_suffix: Suffix used for enhanced directories (default: "_enhanced")
    """
    output_dir = Path(output_dir)
    
    # Check for enhanced directory first if prefer_enhanced
    search_dir = output_dir
    if prefer_enhanced:
        enhanced_dir = output_dir.parent / f"{output_dir.name}{enhanced_suffix}"
        if enhanced_dir.exists() and enhanced_dir.is_dir():
            search_dir = enhanced_dir
            print(f"Using enhanced directory: {search_dir}")
        else:
            print(f"Enhanced directory not found ({enhanced_dir}), using original: {output_dir}")
    
    if not search_dir.exists():
        print(f"Output directory does not exist: {search_dir}")
        return []

    results = []
    subdirs = [d for d in search_dir.iterdir() if d.is_dir() and d.name not in parse_methods]
    if subdirs:
        for subdir in subdirs:
            file_stem = subdir.name
            for method in parse_methods:
                method_dir = subdir / method
                if method_dir.exists():
                    json_file = method_dir / f"{file_stem}_content_list.json"
                    
                    if json_file.exists():
                        is_enhanced = search_dir != output_dir
                        print(f"Found parsed file: {json_file}")
                        print(f"  File name: {file_stem}, Parse method: {method}, Enhanced: {is_enhanced}")

                        content_list = read_content_list_from_json(json_file, method_dir)
                        if content_list:
                            results.append((content_list, file_stem, method, json_file))

    if not results:
        print(f"No parsed files found in directory: {search_dir}")

    return results


def find_all_enhanced_parsed_files(output_dir: Path, parse_methods: list = ["auto", "ocr", "txt", "vlm"]):
    """
    Find all enhanced parsed output files (_content_list_enhanced.json)
    Looks for files with _content_list_enhanced.json suffix in the same directory as original files
    
    Args:
        output_dir: Base output directory (e.g., /path/to/Guidelines/WHO)
        parse_methods: List of parse methods to look for
        
    Returns:
        list: [(content_list, file_stem, method, json_file), ...] list
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return []

    results = []
    subdirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name not in parse_methods]
    
    if subdirs:
        for subdir in subdirs:
            file_stem = subdir.name
            for method in parse_methods:
                method_dir = subdir / method
                if method_dir.exists():
                    # Look for enhanced file first
                    enhanced_json_file = method_dir / f"{file_stem}_content_list_enhanced.json"
                    
                    if enhanced_json_file.exists():
                        print(f"Found enhanced parsed file: {enhanced_json_file}")
                        print(f"  File name: {file_stem}, Parse method: {method}")
                        
                        content_list = read_content_list_from_json(enhanced_json_file, method_dir)
                        if content_list:
                            results.append((content_list, file_stem, method, enhanced_json_file))
                    else:
                        # Fallback to original file if enhanced doesn't exist
                        json_file = method_dir / f"{file_stem}_content_list.json"
                        if json_file.exists():
                            print(f"Found parsed file (not enhanced): {json_file}")
                            print(f"  File name: {file_stem}, Parse method: {method}")
                            print(f"  Warning: Enhanced file not found, using original file")
                            
                            content_list = read_content_list_from_json(json_file, method_dir)
                            if content_list:
                                results.append((content_list, file_stem, method, json_file))

    if not results:
        print(f"No parsed files found in directory: {output_dir}")

    return results


async def fix_invalid_doc_status(rag):
    """Fix invalid doc_status records that are missing required 'status' field"""
    try:
        # Use DocStatus from rag_core
        from pathpocket.rag_core import DocStatus
        
        # Get all doc_status records (this is a workaround since there's no get_all method)
        # We'll check during processing instead
        rag.logger.info("Checking for invalid doc_status records...")
        
        # The fix will happen automatically in _mark_multimodal_processing_complete
        # when it encounters a doc_status without status field
        rag.logger.info("Invalid doc_status records will be fixed automatically during processing")
        
    except Exception as e:
        rag.logger.warning(f"Error fixing invalid doc_status: {e}")


def enhance_content_list(content_list: List[Dict[str, Any]], file_name: str = None) -> List[Dict[str, Any]]:
    """
    Enhance content list:
    1. Add section titles to image/table captions ONLY if caption is empty (previous text_level: 1 items)
       - If the previous text_level: 1 item also has a previous text_level: 1 item, add both
    2. Remove reference sections (text_level: 1 with keywords like "参考文献")
    3. Add file name to the beginning of all non-text item captions (image/table)
    
    Args:
        content_list: Original content list
        file_name: File name to prepend to captions (optional)
        
    Returns:
        Enhanced content list
    """
    enhanced_list = []
    reference_keywords = ["参考文献", "references", "bibliography", "reference"]
    
    # First pass: collect text_level: 1 items and their indices
    text_level_1_indices = []
    for idx, item in enumerate(content_list):
        if item.get("type") == "text" and item.get("text_level") == 1:
            text_level_1_indices.append(idx)
    
    def get_section_titles_before(index: int) -> List[str]:
        """Get section titles (text_level: 1) before the given index
        Returns: List of titles, with the last one first, and its previous one if exists
        Filters out reference section titles (e.g., "参考文献")
        """
        # Find the last non-reference text_level: 1 item before this index
        last_title_idx = None
        for prev_idx in reversed(text_level_1_indices):
            if prev_idx < index:
                title_item = content_list[prev_idx]
                title_text = title_item.get("text", "").strip()
                # Skip reference sections
                if title_text and not any(keyword.lower() in title_text.lower() for keyword in reference_keywords):
                    last_title_idx = prev_idx
                    break
        
        if last_title_idx is None:
            return []
        
        result_titles = []
        # Get the last title text
        last_title_item = content_list[last_title_idx]
        last_title_text = last_title_item.get("text", "").strip()
        if last_title_text:
            result_titles.append(last_title_text)
        
        # # Check if there's a previous non-reference text_level: 1 item before the last one
        # # Find the text_level: 1 item that comes before last_title_idx
        # prev_title_idx = None
        # for prev_idx in reversed(text_level_1_indices):
        #     if prev_idx < last_title_idx:
        #         title_item = content_list[prev_idx]
        #         title_text = title_item.get("text", "").strip()
        #         # Skip reference sections
        #         if title_text and not any(keyword.lower() in title_text.lower() for keyword in reference_keywords):
        #             prev_title_idx = prev_idx
        #             break
        
        # if prev_title_idx is not None:
        #     prev_title_item = content_list[prev_title_idx]
        #     prev_title_text = prev_title_item.get("text", "").strip()
        #     if prev_title_text:
        #         # Insert at the beginning (so order is: previous title, last title)
        #         result_titles.insert(0, prev_title_text)
        
        return result_titles
    
    # Second pass: enhance images/tables and filter out references
    i = 0
    while i < len(content_list):
        item = content_list[i].copy()
        
        # Check if this is a reference section (text_level: 1 with reference keywords)
        if item.get("type") == "text" and item.get("text_level") == 1:
            text = item.get("text", "").strip()
            # Check if text contains reference keywords
            is_reference = any(keyword.lower() in text.lower() for keyword in reference_keywords)
            
            if is_reference:
                # Skip this reference section header and all subsequent items until next text_level: 1
                # or until end of document
                i += 1
                while i < len(content_list):
                    next_item = content_list[i]
                    # Stop at next text_level: 1 item (which would be a new section)
                    if next_item.get("type") == "text" and next_item.get("text_level") == 1:
                        break
                    i += 1
                # If we reached the end, break the outer loop
                if i >= len(content_list):
                    break
                continue
        
        # Enhance image captions - only if caption is empty
        if item.get("type") == "image":
            image_caption = item.get("image_caption", [])
            # Check if caption is empty (None, empty list, or all empty strings)
            is_caption_empty = (
                not image_caption 
                or all(not str(cap).strip() for cap in image_caption)
            )
            
            if is_caption_empty:
                # Only add section titles if caption was empty
                section_titles = get_section_titles_before(i)
                if section_titles:
                    # Use section titles as caption
                    enhanced_caption = " | ".join(section_titles)
                    item["image_caption"] = [enhanced_caption]
            
            # Add file name to the beginning of caption if file_name is provided
            final_caption = item.get("image_caption", [])
            if file_name:
                if final_caption:
                    # Prepend file name to the first caption if not already present
                    first_caption = final_caption[0] if final_caption else ""
                    if first_caption and not first_caption.startswith(file_name):
                        enhanced_caption_with_filename = f"{file_name} | {first_caption}"
                        item["image_caption"] = [enhanced_caption_with_filename] + final_caption[1:]
                    elif not first_caption:
                        # If caption is still empty, use file name as caption
                        item["image_caption"] = [file_name]
                else:
                    # If no caption at all, use file name as caption
                    item["image_caption"] = [file_name]
            
            # Check if caption is still empty after enhancement
            final_caption = item.get("image_caption", [])
            if not final_caption or all(not str(cap).strip() for cap in final_caption):
                # Skip this item (don't add to enhanced_list)
                i += 1
                continue
        
        # Enhance table captions - only if caption is empty
        if item.get("type") == "table":
            table_caption = item.get("table_caption", [])
            # Check if caption is empty (None, empty list, or all empty strings)
            is_caption_empty = (
                not table_caption 
                or all(not str(cap).strip() for cap in table_caption)
            )
            
            if is_caption_empty:
                # Only add section titles if caption was empty
                section_titles = get_section_titles_before(i)
                if section_titles:
                    # Use section titles as caption
                    enhanced_caption = " | ".join(section_titles)
                    item["table_caption"] = [enhanced_caption]
            
            # Add file name to the beginning of caption if file_name is provided
            final_caption = item.get("table_caption", [])
            if file_name:
                if final_caption:
                    # Prepend file name to the first caption if not already present
                    first_caption = final_caption[0] if final_caption else ""
                    if first_caption and not first_caption.startswith(file_name):
                        enhanced_caption_with_filename = f"{file_name} | {first_caption}"
                        item["table_caption"] = [enhanced_caption_with_filename] + final_caption[1:]
                    elif not first_caption:
                        # If caption is still empty, use file name as caption
                        item["table_caption"] = [file_name]
                else:
                    # If no caption at all, use file name as caption
                    item["table_caption"] = [file_name]
            
            # Check if caption is still empty after enhancement
            final_caption = item.get("table_caption", [])
            if not final_caption or all(not str(cap).strip() for cap in final_caption):
                # Skip this item (don't add to enhanced_list)
                i += 1
                continue
        
        # Check text items
        if item.get("type") == "text":
            text = item.get("text", "").strip()
            if not text:
                # Skip empty text items
                i += 1
                continue
        
        enhanced_list.append(item)
        i += 1
    
    return enhanced_list


def should_enhance_folder(folder_name: str) -> bool:
    """
    Determine if folder should be enhanced based on folder name
    
    Args:
        folder_name: Folder name to check
        
    Returns:
        True if should enhance, False otherwise
    """
    # Default: enhance all folders
    # Override this function or use enhance_folders parameter in enhance_parsed_files
    # to specify which folders to enhance
    return True


def enhance_parsed_files(output_dir: Path, enhance_folders: List[str] = None, enhanced_suffix: str = "_enhanced"):
    """
    Enhance parsed JSON files:
    1. Add section titles to image/table captions
    2. Remove reference sections
    
    Args:
        output_dir: Output directory containing parsed files
        enhance_folders: List of folder names to enhance. If None, uses should_enhance_folder to determine.
        enhanced_suffix: Suffix to add to output directory name (default: "_enhanced")
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return
    
    # Check if enhanced directory already exists and contains files
    enhanced_output_dir = output_dir.parent / f"{output_dir.name}{enhanced_suffix}"
    
    # Check if enhanced directory exists and has content
    if enhanced_output_dir.exists() and enhanced_output_dir.is_dir():
        # Check if there are any JSON files in the enhanced directory
        has_enhanced_files = False
        parse_methods = ["auto", "ocr", "txt", "vlm"]
        for subdir in enhanced_output_dir.iterdir():
            if subdir.is_dir() and subdir.name not in parse_methods:
                for method in parse_methods:
                    method_dir = subdir / method
                    if method_dir.exists():
                        json_files = list(method_dir.glob("*_content_list.json"))
                        if json_files:
                            has_enhanced_files = True
                            break
                    if has_enhanced_files:
                        break
            if has_enhanced_files:
                break
        
        if has_enhanced_files:
            print(f"Enhanced directory already exists and contains files: {enhanced_output_dir}")
            print("Skipping enhancement process.")
            return
    
    # Create enhanced output directory
    enhanced_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Enhanced files will be saved to: {enhanced_output_dir}")
    
    parse_methods = ["auto", "ocr", "txt", "vlm"]
    enhanced_count = 0
    skipped_count = 0
    
    # Find all subdirectories
    subdirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name not in parse_methods]
    
    for subdir in subdirs:
        folder_name = subdir.name
        
        # Determine if this folder should be enhanced
        should_enhance = False
        if enhance_folders:
            should_enhance = folder_name in enhance_folders
        else:
            should_enhance = should_enhance_folder(folder_name)
        
        if not should_enhance:
            print(f"Skipping folder (not in enhance list): {folder_name}")
            continue
        
        # Create corresponding directory in enhanced output
        enhanced_subdir = enhanced_output_dir / folder_name
        enhanced_subdir.mkdir(parents=True, exist_ok=True)
        
        # Process each parse method
        for method in parse_methods:
            method_dir = subdir / method
            if not method_dir.exists():
                continue
            
            json_file = method_dir / f"{folder_name}_content_list.json"
            
            # Check if already enhanced (in enhanced output directory)
            enhanced_method_dir = enhanced_subdir / method
            enhanced_method_dir.mkdir(parents=True, exist_ok=True)
            enhanced_file = enhanced_method_dir / f"{folder_name}_content_list.json"
            
            # Skip if already enhanced
            if enhanced_file.exists():
                print(f"Already enhanced, skipping: {enhanced_file}")
                skipped_count += 1
                continue
            
            if not json_file.exists():
                continue
            
            print(f"Enhancing: {json_file}")
            
            try:
                # Read original content list
                content_list = read_content_list_from_json(json_file, method_dir)
                if not content_list:
                    print(f"Warning: Could not read content from {json_file}")
                    continue
                
                # Enhance content list (pass file_name to add it to captions)
                enhanced_content_list = enhance_content_list(content_list, file_name=folder_name)
                
                # Copy images directory if it exists
                images_dir = method_dir / "images"
                if images_dir.exists() and images_dir.is_dir():
                    enhanced_images_dir = enhanced_method_dir / "images"
                    if not enhanced_images_dir.exists():
                        import shutil
                        shutil.copytree(images_dir, enhanced_images_dir)
                        print(f"  Copied images directory to: {enhanced_images_dir}")
                
                # Save enhanced content list
                with open(enhanced_file, "w", encoding="utf-8") as f:
                    json.dump(enhanced_content_list, f, ensure_ascii=False, indent=2)
                
                print(f"✅ Enhanced and saved: {enhanced_file}")
                enhanced_count += 1
                
            except Exception as e:
                print(f"❌ Error enhancing {json_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print(f"Enhancement complete!")
    print(f"  - Enhanced files: {enhanced_count}")
    print(f"  - Skipped files: {skipped_count}")
    print(f"  - Output directory: {enhanced_output_dir}")
    print(f"{'='*60}\n")


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items): Pure text content and multimodal items list
    """
    text_parts = []
    multimodal_items = []

    for item in content_list:
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_items.append(item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = Path(image_path)

        logger.debug(f"Validating image path: {image_path}")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # Check if file exists
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


async def insert_text_content(
    rag_engine,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
):
    """
    Insert pure text content into RAG engine

    Args:
        rag_engine: RAG engine instance
        input: Single document string or list of document strings
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
    """
    logger.info("Starting text content insertion into RAG engine...")

    # Use RAG engine's insert method with all parameters
    await rag_engine.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


async def insert_text_content_with_multimodal_content(
    rag_engine,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    Insert pure text content into RAG engine
    Note: multimodal_content is accepted but not passed to ainsert (it will be processed separately)

    Args:
        rag_engine: RAG engine instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional, will be processed separately)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional, not used by RAG engine)
    """
    # Ensure input is not empty
    if isinstance(input, str):
        if not input.strip():
            logger.warning("Empty text content provided, skipping insertion")
            return
    elif isinstance(input, list):
        # Filter out empty strings
        input = [text for text in input if text and text.strip()]
        if not input:
            logger.warning("No valid text content in input list, skipping insertion")
            return
    
    logger.info("Starting text content insertion into RAG engine...")

    # Use RAG engine's insert method (without multimodal_content and scheme_name as they're not supported)
    # Multimodal content will be processed separately by _process_multimodal_content
    await rag_engine.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")
