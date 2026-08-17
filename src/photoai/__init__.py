"""photo-ai-toolkit, as one package.

Every module here imports its siblings absolutely -- `from photoai import
scoring` -- so that installing the project puts one name on `sys.path` instead
of fifty.
"""
