# Data Pipeline Module Documentation

## Overview

The data pipeline module handles document upload, parsing, and processing for RAG (Retrieval-Augmented Generation). It supports multiple file formats and provides comprehensive file validation and processing capabilities.

## Files

### `file_service.py`
Main file upload and processing service with comprehensive file handling.

**Key Features:**
- File upload and validation
- Multiple file format support
- RAG initialization
- Error handling and logging

**Usage:**
```python
from blackbird_sdk.data_pipeline.file_service import FileService

file_service = FileService(http_client, logger)

# Upload single file
response = file_service.upload_single_file(
    "financial_report.pdf", 
    "finance"
)

# Upload multiple files
response = file_service.upload_multiple_files(
    ["report1.pdf", "report2.xlsx"], 
    "finance"
)
```

### `pdf_parser.py`
PDF document parsing and text extraction.

**Key Features:**
- PDF text extraction
- Table extraction
- Image extraction
- Metadata extraction

**Usage:**
```python
from blackbird_sdk.data_pipeline.pdf_parser import PDFParser

parser = PDFParser()
content = parser.parse("document.pdf")
```

### `excel_parser.py`
Excel and CSV file parsing.

**Key Features:**
- Excel file parsing
- CSV file parsing
- Table data extraction
- Formula evaluation

**Usage:**
```python
from blackbird_sdk.data_pipeline.excel_parser import ExcelParser

parser = ExcelParser()
data = parser.parse("financial_data.xlsx")
```

### `docx_parser.py`
Microsoft Word document parsing.

**Key Features:**
- DOCX file parsing
- Text extraction
- Table extraction
- Formatting preservation

**Usage:**
```python
from blackbird_sdk.data_pipeline.docx_parser import DocxParser

parser = DocxParser()
content = parser.parse("document.docx")
```

### `markdown_parser.py`
Markdown file parsing and processing.

**Key Features:**
- Markdown parsing
- Code block extraction
- Link extraction
- Metadata parsing

**Usage:**
```python
from blackbird_sdk.data_pipeline.markdown_parser import MarkdownParser

parser = MarkdownParser()
content = parser.parse("README.md")
```

### `ocr.py`
Optical Character Recognition for image-based documents.

**Key Features:**
- Image text extraction
- Handwriting recognition
- Table recognition
- Layout analysis

**Usage:**
```python
from blackbird_sdk.data_pipeline.ocr import OCRParser

parser = OCRParser()
text = parser.extract_text("scanned_document.jpg")
```

### `pipeline.py`
Document processing pipeline for complex workflows.

**Key Features:**
- Multi-step processing
- Pipeline configuration
- Error recovery
- Progress tracking

**Usage:**
```python
from blackbird_sdk.data_pipeline.pipeline import DocumentPipeline

pipeline = DocumentPipeline()
result = pipeline.process("document.pdf")
```

### `base.py`
Base parser class for extensibility.

**Key Features:**
- Common parser interface
- Error handling
- Configuration management
- Extensibility hooks

**Usage:**
```python
from blackbird_sdk.data_pipeline.base import BaseParser

class CustomParser(BaseParser):
    def parse(self, file_path):
        # Custom parsing logic
        return content
```

## Supported File Types

### Documents
- **PDF** (.pdf): Portable Document Format
- **DOCX** (.docx): Microsoft Word documents
- **TXT** (.txt): Plain text files
- **MD** (.md): Markdown files

### Spreadsheets
- **XLSX** (.xlsx): Microsoft Excel files
- **XLS** (.xls): Legacy Excel files
- **CSV** (.csv): Comma-separated values

### Code Files
- **PY** (.py): Python files
- **JS** (.js): JavaScript files
- **HTML** (.html): HTML files
- **CSS** (.css): CSS files
- **JSON** (.json): JSON data files
- **XML** (.xml): XML files
- **YAML** (.yml, .yaml): YAML configuration files

### Audio Files
- **WAV** (.wav): Wave audio files
- **MP3** (.mp3): MP3 audio files
- **M4A** (.m4a): Apple audio files
- **FLAC** (.flac): Lossless audio files

### Image Files
- **JPG/JPEG** (.jpg, .jpeg): JPEG images
- **PNG** (.png): PNG images
- **GIF** (.gif): GIF images
- **BMP** (.bmp): Bitmap images
- **TIFF** (.tiff): TIFF images

## Agent-Specific File Support

### Finance Agent
**Supported Formats:**
- Documents: PDF, DOCX, TXT, MD
- Spreadsheets: XLSX, XLS, CSV

**Use Cases:**
- Financial reports analysis
- Spreadsheet data processing
- Investment analysis
- Accounting documents

### Legal Agent
**Supported Formats:**
- Documents: PDF, DOCX, TXT, MD

**Use Cases:**
- Contract analysis
- Legal document review
- Compliance checking
- Legal research

### Tech Agent
**Supported Formats:**
- Documents: PDF, DOCX, TXT, MD
- Code: PY, JS, HTML, CSS, JSON, XML, YAML

**Use Cases:**
- Code review
- Technical documentation
- API documentation
- Configuration files

### Meetings Agent
**Supported Formats:**
- Documents: PDF, DOCX, TXT, MD
- Audio: WAV, MP3, M4A, FLAC

**Use Cases:**
- Meeting transcription
- Audio analysis
- Meeting notes
- Action item extraction

### Research Agent
**Supported Formats:**
- Documents: PDF, DOCX, TXT, MD

**Use Cases:**
- Research papers
- Academic documents
- Literature review
- Information synthesis

### Image-generator Agent
**Supported Formats:**
- Images: JPG, PNG, GIF, BMP, TIFF
- Documents: PDF, DOCX, TXT, MD

**Use Cases:**
- Image analysis
- Style transfer
- Content creation
- Design inspiration

## File Processing Workflow

### 1. File Validation
```python
# Validate file before upload
is_valid, message = file_service.validate_file("document.pdf", "finance")
if not is_valid:
    print(f"Validation failed: {message}")
```

### 2. File Upload
```python
# Upload single file
response = sdk.upload_file("financial_report.pdf", "finance")

# Upload multiple files
files = ["report1.pdf", "report2.xlsx", "data.csv"]
response = sdk.upload_files(files, "finance")
```

### 3. RAG Initialization
```python
# Initialize agent with files
sdk.initialize_agent_with_files("finance", ["report.pdf"])

# Send message with file context
response = sdk.send_message("Analyze this financial data")
```

## File Validation

### Size Limits
- **Maximum file size**: 100MB per file
- **Total upload size**: 500MB per session
- **File count limit**: 50 files per session

### Format Validation
```python
# Check supported formats for agent
supported_formats = file_service.get_supported_extensions_for_agent("finance")
print(f"Supported formats: {supported_formats}")
```

### Content Validation
```python
# Get file information
file_info = file_service.get_file_info("document.pdf")
print(f"File size: {file_info['size']} bytes")
print(f"File type: {file_info['type']}")
```

## Error Handling

### Common Errors
1. **FileProcessingError**: File upload/processing issues
2. **ValidationError**: File validation failures
3. **UnsupportedFormatError**: Unsupported file format
4. **FileSizeError**: File too large
5. **PermissionError**: File access denied

### Error Handling Example
```python
try:
    response = sdk.upload_file("document.pdf", "finance")
except FileProcessingError as e:
    print(f"File processing failed: {e}")
    print(f"File: {e.file_path}")
    print(f"Type: {e.file_type}")
except ValidationError as e:
    print(f"Validation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### File Processing
- **Parallel processing**: Multiple files processed concurrently
- **Caching**: Processed content cached for reuse
- **Compression**: Large files compressed during upload
- **Streaming**: Large files streamed in chunks

### Memory Management
- **Chunked processing**: Large files processed in chunks
- **Garbage collection**: Automatic cleanup of processed files
- **Memory limits**: Configurable memory usage limits

## Testing

### Unit Tests
```python
# Test file validation
def test_file_validation():
    file_service = FileService(http_client, logger)
    is_valid, message = file_service.validate_file("test.pdf", "finance")
    assert is_valid == True

# Test file upload
def test_file_upload():
    sdk = BlackbirdSDK()
    response = sdk.upload_file("test.pdf", "finance")
    assert response is not None
```

### Integration Tests
```python
# Test multiple file upload
def test_multiple_files():
    sdk = BlackbirdSDK()
    files = ["test1.pdf", "test2.xlsx"]
    response = sdk.upload_files(files, "finance")
    assert response is not None

# Test agent with files
def test_agent_with_files():
    sdk = BlackbirdSDK()
    sdk.initialize_agent_with_files("finance", ["test.pdf"])
    response = sdk.send_message("Analyze this")
    assert response is not None
```

## Configuration

### File Service Configuration
```python
config = {
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "max_total_size": 500 * 1024 * 1024,  # 500MB
    "max_file_count": 50,
    "supported_formats": ["pdf", "docx", "xlsx"],
    "enable_compression": True,
    "enable_caching": True
}

file_service = FileService(http_client, logger, config)
```

### Parser Configuration
```python
parser_config = {
    "extract_tables": True,
    "extract_images": False,
    "preserve_formatting": True,
    "ocr_enabled": True
}

parser = PDFParser(parser_config)
```

## Best Practices

1. **File Validation**: Always validate files before upload
2. **Error Handling**: Handle file processing errors gracefully
3. **Format Selection**: Use appropriate formats for your use case
4. **Size Management**: Monitor file sizes and total upload limits
5. **Caching**: Enable caching for frequently used files

## Troubleshooting

### Common Issues

1. **File upload fails**
   - Check file size limits
   - Verify file format is supported
   - Check file permissions
   - Verify network connectivity

2. **Processing errors**
   - Check file corruption
   - Verify file format compatibility
   - Check available memory
   - Review error logs

3. **Slow processing**
   - Enable compression
   - Use appropriate file formats
   - Check system resources
   - Enable caching

### Debug Mode
```python
# Enable debug logging
sdk = BlackbirdSDK(development_mode=True)

# Check file processing logs
file_service.logger.setLevel("DEBUG")
```

## API Reference

### FileService
- `upload_single_file(file_path, agent_type, options=None)`: Upload single file
- `upload_multiple_files(file_paths, agent_type, options=None)`: Upload multiple files
- `validate_file(file_path, agent_type=None)`: Validate file
- `get_supported_extensions_for_agent(agent_type)`: Get supported formats
- `get_file_info(file_path)`: Get file information

### BaseParser
- `parse(file_path)`: Parse file content
- `validate(file_path)`: Validate file format
- `extract_metadata(file_path)`: Extract file metadata

### Configuration Options
- `max_file_size`: Maximum file size in bytes
- `max_total_size`: Maximum total upload size
- `max_file_count`: Maximum number of files
- `supported_formats`: List of supported formats
- `enable_compression`: Enable file compression
- `enable_caching`: Enable result caching 