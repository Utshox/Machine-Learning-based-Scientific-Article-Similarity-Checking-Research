import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def create_mock_pan_dataset(base_path):
    # 1. Setup Folder Structure
    susp_dir = os.path.join(base_path, 'susp')
    src_dir = os.path.join(base_path, 'src')
    os.makedirs(susp_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    # Sample texts for generation
    source_content = "This is the original scientific research about machine learning and its impact on modern data analysis. " \
                     "Data scientists use various algorithms to detect patterns in big data. " \
                     "The conclusion of this study suggests that hybrid models are superior for similarity checking."
    
    unrelated_content = "The biology of the Amazon rainforest is very diverse. " \
                        "Many species of plants and animals live in this unique ecosystem. "
    
    ai_generated_style = "Artificial Intelligence has transformed the way we write academic papers. " \
                         "Large Language Models can now generate text that sounds very human-like."

    # --- Scenario 1: Plagiarism Case ---
    # Source Document
    src_filename = "source-document001.txt"
    with open(os.path.join(src_dir, src_filename), "w") as f:
        f.write(source_content)

    # Suspicious Document (Unrelated + Plagiarized + AI Altered)
    susp_filename = "suspicious-document001.txt"
    plagiarized_part = "The study's conclusion indicates that hybrid models are better for checking similarity."
    
    susp_full_text = unrelated_content + plagiarized_part + " " + ai_generated_style
    
    with open(os.path.join(susp_dir, susp_filename), "w") as f:
        f.write(susp_full_text)

    # 2. Create XML Truth File
    root = ET.Element("document", reference=susp_filename)
    
    # Calculate Offsets
    this_offset = len(unrelated_content)
    this_length = len(plagiarized_part)
    source_offset = source_content.find("The conclusion of this study suggests that hybrid models are superior for similarity checking.")
    source_length = len("The conclusion of this study suggests that hybrid models are superior for similarity checking.")
    
    # Plagiarism Feature
    ET.SubElement(root, "feature", 
                  name="plagiarism", 
                  type="manual", 
                  this_offset=str(this_offset), 
                  this_length=str(this_length),
                  source_reference=src_filename,
                  source_offset=str(source_offset),
                  source_length=str(source_length))
    
    # Altered Feature (AI Generated but not plagiarized)
    alt_offset = susp_full_text.find(ai_generated_style)
    alt_length = len(ai_generated_style)
    ET.SubElement(root, "feature", 
                  name="altered", 
                  type="artificial", 
                  this_offset=str(alt_offset), 
                  this_length=str(alt_length))

    truth_filename = susp_filename.replace(".txt", ".xml")
    with open(os.path.join(susp_dir, truth_filename), "w") as f:
        f.write(prettify(root))

    # 3. Create Pairs file
    with open(os.path.join(base_path, "pairs"), "w") as f:
        f.write(f"{susp_filename} {src_filename}\n")

    print(f"Mock PAN dataset created at: {base_path}")

if __name__ == "__main__":
    base_data_path = "/Users/shinthiya.promi/Desktop/MS_THESIS/implementation/data/pan2025_mock"
    create_mock_pan_dataset(base_data_path)
