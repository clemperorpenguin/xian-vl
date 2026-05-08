import sqlite3
import json
from pathlib import Path

class JX3Database:
    def __init__(self, data_dir: str = None, localization_file: str = None):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        project_root = Path(__file__).parent.parent.parent
        self.data_dir = Path(data_dir) if data_dir else project_root / "xian" / "knowledge" / "jx3box-data" / "data"
        self.localization_file = Path(localization_file) if localization_file else project_root / "xian" / "knowledge" / "localization.json"
        
        self._init_schema()
        self._load_data()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY,
                cn_name TEXT UNIQUE,
                en_name TEXT,
                role TEXT,
                description TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS specs (
                id INTEGER PRIMARY KEY,
                cn_name TEXT UNIQUE,
                en_name TEXT,
                school_id INTEGER,
                role TEXT,
                type TEXT,
                FOREIGN KEY (school_id) REFERENCES classes(id)
            )
        ''')
        self.conn.commit()

    def _load_data(self):
        # Load localizations
        localizations = {"classes": {}, "specs": {}}
        if self.localization_file.exists():
            with open(self.localization_file, "r", encoding="utf-8") as f:
                localizations = json.load(f)

        cursor = self.conn.cursor()

        # Load Classes (Schools)
        school_json = self.data_dir / "xf" / "school.json"
        if school_json.exists():
            with open(school_json, "r", encoding="utf-8") as f:
                schools = json.load(f)
                for cn_name, data in schools.items():
                    loc = localizations["classes"].get(cn_name, {})
                    cursor.execute('''
                        INSERT OR IGNORE INTO classes (id, cn_name, en_name, role, description)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        data["school_id"], 
                        cn_name, 
                        loc.get("en", cn_name),
                        loc.get("role", ""),
                        loc.get("desc", "")
                    ))

        # Load Specs (XinFa)
        xf_json = self.data_dir / "xf" / "xf.json"
        if xf_json.exists():
            with open(xf_json, "r", encoding="utf-8") as f:
                specs = json.load(f)
                for cn_name, data in specs.items():
                    if cn_name == "通用" or data.get("id") == 0:
                        continue
                    loc = localizations["specs"].get(cn_name, {})
                    cursor.execute('''
                        INSERT OR IGNORE INTO specs (id, cn_name, en_name, school_id, role, type)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        data["id"], 
                        cn_name, 
                        loc.get("en", cn_name),
                        data["school"],
                        loc.get("role", ""),
                        loc.get("type", "")
                    ))

        self.conn.commit()

    def get_classes(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM classes")
        return [dict(row) for row in cursor.fetchall()]

    def get_class_by_name(self, name: str):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM classes WHERE cn_name = ? OR en_name = ? COLLATE NOCASE", (name, name))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_specs_by_class(self, class_id: int):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM specs WHERE school_id = ?", (class_id,))
        return [dict(row) for row in cursor.fetchall()]

    def search_spec(self, query: str):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM specs WHERE cn_name LIKE ? OR en_name LIKE ? COLLATE NOCASE", (f"%{query}%", f"%{query}%"))
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()

# Singleton instance
db = None
def get_db():
    global db
    if db is None:
        db = JX3Database()
    return db
