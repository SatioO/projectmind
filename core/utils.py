import re
import uuid
import unicodedata


def sanitize_filename(filename, max_length=255):
    """
      Robust filename sanitizer that ensures the filename is safe for any OS, URL, or database usage.

      This function performs the following transformations:
          1. Normalize Unicode characters (e.g., accented letters) to a standard form.
          2. Remove all unsafe characters for filesystems and URLs.
          3. Replace spaces and other whitespace characters with underscores.
          4. Keep only letters, numbers, hyphens, underscores, and dots (for file extensions).
          5. Optionally truncate the filename if it exceeds `max_length` characters.

      Parameters:
          filename (str): The original filename provided by the user.
          max_length (int, optional): Maximum allowed length of the sanitized filename.
                                      Defaults to 255 characters.

      Returns:
          str: A sanitized, safe filename suitable for storage or URL usage.

      Example:
          >>> sanitize_filename_strict("Screenshot 2026-03-24 at 11.31.59 AM (1).png")
          'Screenshot_2026-03-24_at_11.31.59_AM_1.png'
    """

    # Normalize Unicode to standard form
    filename = unicodedata.normalize("NFKD", filename)

    # Replace spaces and non-breaking/tricky spaces with underscore
    filename = re.sub(r'\s+', '_', filename)

    # Remove all characters except letters, numbers, hyphen, underscore, and dot
    filename = re.sub(r'[^A-Za-z0-9._-]', '', filename)

    # Avoid filenames starting or ending with a dot
    filename = filename.strip('.')

    # Truncate filename to max_length (keeping extension if present)
    if len(filename) > max_length:
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:  # has extension
            name, ext = name_parts
            name = name[:max_length - len(ext) - 1]
            filename = f"{name}.{ext}"
        else:
            filename = filename[:max_length]

    return filename


def generate_doc_id(project_id: str, filename: str):
    # Generate deterministic namespace UUID from project_id string
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, project_id)
    # Generate deterministic UUID for this filename
    document_uuid = uuid.uuid5(namespace, filename)
    return document_uuid
