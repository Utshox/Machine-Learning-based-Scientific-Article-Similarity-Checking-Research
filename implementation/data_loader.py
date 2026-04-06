import os
import re
import xml.etree.ElementTree as ET
import zipfile


PAIR_PATTERN = re.compile(
    r"(suspicious-document\d+)\-(source-document\d+)\.xml$"
)

class PANDataLoader:
    """
    Handles loading of PAN dataset files (txt and XML truths).
    """
    def __init__(self, base_path):
        self.base_path = base_path
        self.is_zip = zipfile.is_zipfile(base_path)
        self.archive = zipfile.ZipFile(base_path) if self.is_zip else None
        self._pairs = None

        if self.is_zip:
            self.members = self.archive.namelist()
            self.susp_dir = self._find_member_dir("/susp/")
            self.src_dir = self._find_member_dir("/src/")
            self.truth_dir = self._find_member_dir("_truth/")
            self.pairs_file = self._find_member("pairs")
        else:
            self.susp_dir = os.path.join(base_path, "susp")
            self.src_dir = os.path.join(base_path, "src")
            self.pairs_file = os.path.join(base_path, "pairs")
            self.truth_dir = self._resolve_truth_dir()

    def _resolve_truth_dir(self):
        direct_truth_dir = os.path.join(self.base_path, "truth")
        if os.path.isdir(direct_truth_dir):
            return direct_truth_dir

        sibling_name = os.path.basename(self.base_path.rstrip(os.sep)) + "_truth"
        sibling_truth_dir = os.path.join(os.path.dirname(self.base_path), sibling_name)
        if os.path.isdir(sibling_truth_dir):
            return sibling_truth_dir

        return self.susp_dir

    def _find_member_dir(self, marker):
        for member in self.members:
            if marker in member:
                return (member.split(marker)[0] + marker).rstrip("/")
        return None

    def _find_member(self, suffix):
        for member in self.members:
            if member.endswith(suffix):
                return member
        return None

    def _read_text_file(self, path):
        if self.is_zip:
            with self.archive.open(path) as handle:
                return handle.read().decode("utf-8")
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def get_pairs(self):
        """Returns a list of (suspicious_file, source_file) tuples."""
        if self._pairs is not None:
            return self._pairs

        pairs = []
        if self.is_zip:
            if self.pairs_file:
                for line in self._read_text_file(self.pairs_file).splitlines():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        pairs.append((parts[0], parts[1]))
            elif self.truth_dir:
                for member in self.members:
                    if not member.startswith(self.truth_dir) or not member.endswith(".xml"):
                        continue
                    match = PAIR_PATTERN.search(os.path.basename(member))
                    if match:
                        pairs.append((f"{match.group(1)}.txt", f"{match.group(2)}.txt"))
        else:
            if os.path.exists(self.pairs_file):
                with open(self.pairs_file, "r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            pairs.append((parts[0], parts[1]))
            elif os.path.isdir(self.truth_dir):
                for filename in sorted(os.listdir(self.truth_dir)):
                    match = PAIR_PATTERN.search(filename)
                    if match:
                        pairs.append((f"{match.group(1)}.txt", f"{match.group(2)}.txt"))
            else:
                self._pairs = pairs
                return pairs

        self._pairs = pairs
        return pairs

    def load_text(self, filename, is_suspicious=True):
        """Loads the content of a text file."""
        folder = self.susp_dir if is_suspicious else self.src_dir
        if self.is_zip:
            path = f"{folder}/{filename}"
        else:
            path = os.path.join(folder, filename)
        return self._read_text_file(path)

    def get_text_size(self, filename, is_suspicious=True):
        """Returns the compressed member size or on-disk file size for a text file."""
        folder = self.susp_dir if is_suspicious else self.src_dir
        if self.is_zip:
            path = f"{folder}/{filename}"
            return self.archive.getinfo(path).file_size

        path = os.path.join(folder, filename)
        return os.path.getsize(path)

    def load_truth(self, susp_filename, src_filename=None):
        """Parses the XML truth file for a suspicious document."""
        if self.is_zip:
            if not self.truth_dir:
                return []

            if src_filename:
                xml_filename = (
                    f"{susp_filename.replace('.txt', '')}-"
                    f"{src_filename.replace('.txt', '')}.xml"
                )
            else:
                xml_filename = f"{susp_filename.replace('.txt', '')}.xml"
            path = f"{self.truth_dir}/{xml_filename}"
            if path not in self.members:
                return []
            xml_content = self._read_text_file(path)
            root = ET.fromstring(xml_content)
        else:
            if src_filename:
                xml_filename = (
                    f"{susp_filename.replace('.txt', '')}-"
                    f"{src_filename.replace('.txt', '')}.xml"
                )
            else:
                xml_filename = susp_filename.replace(".txt", ".xml")
            path = os.path.join(self.truth_dir, xml_filename)
            if not os.path.exists(path):
                return []
            tree = ET.parse(path)
            root = tree.getroot()

        truths = []
        for feature in root.findall("feature"):
            if feature.get("name") == "plagiarism":
                truths.append({
                    "this_offset": int(feature.get("this_offset")),
                    "this_length": int(feature.get("this_length")),
                    "source_reference": feature.get("source_reference"),
                    "source_offset": int(feature.get("source_offset")),
                    "source_length": int(feature.get("source_length"))
                })
        return truths
