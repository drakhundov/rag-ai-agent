from pathlib import Path

import pytest

from ragsuite.ingestion import DocFmt, FileSystemReader


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SAMPLE_DATA_FILES = sorted(
	[
		path
		for path in DATA_DIR.iterdir()
		if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
	],
	key=lambda path: path.name,
)


def _document_signature(docs):
	return [(doc.page_content, doc.metadata.get("source")) for doc in docs]


def test_load_from_fs_returns_all_docs_for_all_sample_files():
	loader = FileSystemReader()
	paths = [
		str(path) if index % 2 else path
		for index, path in enumerate(SAMPLE_DATA_FILES)
	]

	expected_docs = []
	for path in SAMPLE_DATA_FILES:
		expected_docs.extend(loader._load_doc_from_fs(path))

	loaded_docs = loader.load(paths)

	assert len(loaded_docs) == len(expected_docs)
	assert _document_signature(loaded_docs) == _document_signature(expected_docs)


@pytest.mark.parametrize("allowed_fmts", [
	[".txt"],
	[DocFmt.TXT],
	[DocFmt.TXT.value],
	[".pdf"],
	[DocFmt.PDF],
	[DocFmt.PDF.value],
	[DocFmt.TXT, DocFmt.PDF],
	[DocFmt.TXT.value, DocFmt.PDF.value],
])
def test_allowed_formats_accept_expected_real_files_and_reject_the_others(allowed_fmts):
	loader = FileSystemReader(allowed_fmts=allowed_fmts)

	allowed_suffixes = {fmt.value if isinstance(fmt, DocFmt) else str(fmt) for fmt in allowed_fmts}

	for file_path in SAMPLE_DATA_FILES:
		if file_path.suffix.lower() in allowed_suffixes:
			docs = loader._load_doc_from_fs(file_path)
			assert docs
			assert all(doc.metadata.get("source") == str(file_path) for doc in docs)
		else:
			with pytest.raises(ValueError, match="is not allowed in this object"):
				loader._load_doc_from_fs(file_path)


@pytest.mark.parametrize(
	"suffix, content, allowed_fmt",
	[
		(".md", "# heading\n\nmarkdown content", DocFmt.MD),
		(".html", "<html><body>html content</body></html>", DocFmt.HTML),
	],
)
def test_text_like_formats_are_loaded_when_explicitly_allowed(tmp_path, suffix, content, allowed_fmt):
	file_path = tmp_path / f"sample{suffix}"
	file_path.write_text(content, encoding="utf-8")

	loader = FileSystemReader(allowed_fmts=[allowed_fmt])

	docs = loader._load_doc_from_fs(file_path)

	assert len(docs) == 1
	assert docs[0].page_content == content
	assert docs[0].metadata["source"] == str(file_path)


def test_text_file_is_loaded_verbatim_from_data_directory():
	loader = FileSystemReader()
	text_file = DATA_DIR / "text.txt"

	docs = loader._load_doc_from_fs(text_file)

	assert len(docs) == 1
	assert docs[0].page_content == text_file.read_text(encoding="utf-8")
	assert docs[0].metadata["source"] == str(text_file)


@pytest.mark.parametrize("file_path", [path for path in SAMPLE_DATA_FILES if path.suffix.lower() == ".pdf"])
def test_pdf_files_are_loaded_with_source_metadata(file_path):
	loader = FileSystemReader()

	docs = loader._load_doc_from_fs(file_path)

	assert docs
	assert all(doc.metadata["source"] == str(file_path) for doc in docs)


def test_load_from_fs_returns_empty_list_for_empty_input():
	loader = FileSystemReader()

	assert loader.load([]) == []


def test_missing_path_raises_for_public_and_private_loaders(tmp_path):
	loader = FileSystemReader()
	missing_path = tmp_path / "missing.txt"

	with pytest.raises(FileNotFoundError, match="couldn't locate path"):
		loader.load([missing_path])

	with pytest.raises(FileNotFoundError, match="couldn't locate path"):
		loader._load_doc_from_fs(missing_path)
