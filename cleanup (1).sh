#!/usr/bin/env bash
# Run once from the accformula-main folder (git repo root). Removes stray / duplicate files; nothing here is referenced by the app.
set -e
git rm --cached .env 2>/dev/null || true        # stop tracking the committed .env; your local copy stays (it is now in .gitignore)
rm -f "supabase/functions/design-generator/Index."
rm -f "src/components/AssemblyScene (2).tsx"
rm -f "supabase/migrations/20261102120000_add_project_assemblies (1).sql"   # replaced by 20261102120000_add_project_assemblies.sql
echo "frontend cleanup done"
