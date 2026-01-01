# Security Rules

These rules apply to ALL work in this project.

## Credentials
- NEVER commit credentials, API keys, or secrets
- NEVER edit .env files without explicit permission
- NEVER read or expose credential file contents

## File Safety
- NEVER delete files without explicit permission
- NEVER create files in system directories
- ALWAYS ask before creating new top-level directories

## Data Protection
- Treat user data files (in data/, outputs/) as sensitive
- Don't commit data files to git
- Don't log or print file contents that might contain sensitive data

## Dependencies
- Don't add dependencies without updating pyproject.toml
- Verify dependency sources before adding
- Prefer well-maintained packages with active security updates
