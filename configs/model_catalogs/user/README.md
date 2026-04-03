# User Catalogs

This folder is for writable runtime state and personal catalogs.

Typical files here include:
- `user_local_models.catalog.json`
- `unregistered_install_catalog.catalog.json`
- personal catalogs such as `my_catalog.catalog.json`
- `installed_model_links.json`

These files are runtime/user state and should not be treated as committed preset catalogs.

Committed preset catalogs and reference catalogs belong in the parent `configs/model_catalogs` folder.

