import os
import unittest

this_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(
    this_dir,
    os.path.join(
        os.pardir,
        os.path.join(os.pardir, "docs")
    )
)
HAS_DOCS = os.path.isdir(docs_dir)

print(docs_dir)
print(HAS_DOCS)

class DocumentationTutorialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tutorial_src_path = docs_dir

    @unittest.skipIf(not HAS_DOCS, "Docs directory not found, cannot test tutorial")
    def test_tutorial_valid(self):
        pass

    def test_code(self):
        pass