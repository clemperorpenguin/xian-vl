from xian.knowledge.jx3_db import get_db

def query_jx3_database(query: str) -> str:
    """
    Search the JX3 database for a class or spec name and return detailed information.
    """
    db = get_db()
    
    # Check if query matches a class
    cls = db.get_class_by_name(query)
    if cls:
        specs = db.get_specs_by_class(cls['id'])
        specs_str = "\n".join([f"  - {s['cn_name']} ({s['en_name']}): {s['role']} [{s['type']}]" for s in specs])
        return (f"Class: {cls['cn_name']} ({cls['en_name']})\n"
                f"Role: {cls['role']}\n"
                f"Description: {cls['description']}\n"
                f"Specs (XinFa):\n{specs_str}")
                
    # Check if query matches a spec
    specs = db.search_spec(query)
    if specs:
        results = []
        for s in specs:
            # find its class
            classes = db.get_classes()
            cls_name = "Unknown"
            for c in classes:
                if c['id'] == s['school_id']:
                    cls_name = f"{c['cn_name']} ({c['en_name']})"
                    break
            
            results.append(f"Spec: {s['cn_name']} ({s['en_name']})\n"
                           f"Class: {cls_name}\n"
                           f"Role: {s['role']} [{s['type']}]")
        return "\n\n".join(results)
        
    return f"No results found in JX3 database for '{query}'."
