import json
import re
import networkx as nx
from pathlib import Path

from mvaa.utils.graph import export_graph
from mvaa.utils.embeddings import embed_texts
from mvaa.utils.groq_client import get_client

CREATE_TABLE_RE = re.compile(
    r"create table (\w+)\s*\((.*?)\);",
    re.IGNORECASE | re.DOTALL
)

COLUMN_RE = re.compile(
    r"^\s*(\w+)\s+([\w()]+)",
    re.IGNORECASE
)

PK_RE = re.compile(
    r"primary key\s*\((.*?)\)",
    re.IGNORECASE
)

FK_RE = re.compile(
    r"foreign key\s*\((.*?)\)\s*references\s*(\w+)\s*\((.*?)\)",
    re.IGNORECASE
)

SYSTEM_PROMPT = (
    "You generate concise, neutral semantic descriptions of domain entities "
    "represented by database schemas. "
    "Describe the entity (do not mention it is an entity, we already know) and the main types of information it stores. "
    "Mention relationships to other entities when foreign keys are present, "
    "but do not name specific columns, identifiers, or keys. "
    "Avoid generic phrases such as 'collection of', 'attributes', or "
    "'various information'. "
    "Avoid mentioning schema or entity names or using uppercase identifiers. "
    "Use direct, nominal phrasing. "
    "Do not infer behavior, business rules, or processes. "
    "Do not mention SQL, tables, columns, or implementation details. "
    "Produce exactly one concise sentence."
)

COLUMN_SYSTEM_PROMPT = (
    "You generate concise, neutral semantic descriptions of domain attributes "
    "based strictly on structural information. "
    "Describe what the attribute represents in the domain and the entity it belongs to. "
    "If the attribute represents a reference to another entity, mention the association. "
    "Do not assume uniqueness or identifier semantics unless explicitly specified. "
    "Do not infer behavior, business rules, or processes. "
    "Do not mention SQL, tables, columns, data types, or implementation details. "
    "Use direct, nominal phrasing. "
    "Produce exactly one short sentence."
)




USER_TEMPLATE = """
Given the following database schema, produce a single-sentence description
of the domain entity represented and the kind of information it stores.

Schema:
{schema_json}
"""

COLUMN_USER_TEMPLATE = """
Given the following attribute description, produce a single-sentence
semantic description of what the attribute represents.

Attribute:
{attribute_json}
"""

def build_data_graph(sql_path: str):
    G = nx.MultiDiGraph()
    sql_text = Path(sql_path).read_text()

    pending_fks = []

    for table, body in CREATE_TABLE_RE.findall(sql_text):
        table = table.upper()
        table_id = f"table:{table}"

        G.add_node(
            table_id,
            id=table_id,
            nombre=table,
            tipo="tabla",
            descripcion="",
            origen=sql_path
        )

        pk_columns = set()
        lines = [l.strip() for l in body.split(",")]

        for line in lines:
            if line.lower().startswith("constraint"):
                continue

            m = COLUMN_RE.match(line)
            if not m:
                continue

            col, col_type = m.groups()
            col = col.upper()

            col_id = f"{table}.{col}"

            G.add_node(
                col_id,
                id=col_id,
                nombre=col,
                tabla=table,
                tipo="columna",
                sql_type=col_type,
                pk=False,
                descripcion="",
                origen=sql_path
            )

            # table -> column
            G.add_edge(
                table_id,
                col_id,
                tipo="contiene"
            )

        for line in lines:
            m = PK_RE.search(line)
            if not m:
                continue

            for pk in m.group(1).split(","):
                pk_columns.add(pk.strip().upper())

        for pk in pk_columns:
            col_id = f"{table}.{pk}"
            if col_id in G.nodes:
                G.nodes[col_id]["pk"] = True

        for line in lines:
            m = FK_RE.search(line)
            if not m:
                continue

            src_cols, ref_table, ref_cols = m.groups()
            ref_table = ref_table.upper()

            src_cols = [c.strip().upper() for c in src_cols.split(",")]
            ref_cols = [c.strip().upper() for c in ref_cols.split(",")]

            for src, ref in zip(src_cols, ref_cols):
                src_id = f"{table}.{src}"
                dst_id = f"{ref_table}.{ref}"

                pending_fks.append(
                    (src_id, dst_id, line.strip())
                )


    for src_id, dst_id, origen in pending_fks:
        if src_id in G.nodes and dst_id in G.nodes:
            G.add_edge(
                src_id,
                dst_id,
                tipo="FK",
                origen=origen
            )

    return G

def extract_table_facts(G, table_node):
    table_name = G.nodes[table_node]["nombre"]

    columns = []
    pk_columns = []
    referenced_tables = set()

    for _, col_node, edata in G.out_edges(table_node, data=True):
        if edata.get("tipo") != "contiene":
            continue

        col_data = G.nodes[col_node]
        columns.append(col_data["nombre"])

        if col_data.get("pk"):
            pk_columns.append(col_data["nombre"])

        # FK: columna -> columna
        for _, dst, fk_data in G.out_edges(col_node, data=True):
            if fk_data.get("tipo") == "FK":
                ref_table = dst.split(".")[0]
                referenced_tables.add(ref_table)

    return {
        "table": table_name,
        "columns": sorted(columns),
        "primary_keys": sorted(pk_columns),
        "references": sorted(referenced_tables),
    }


def generate_table_descriptor(table_name, columns, foreign_keys):
    schema = {
        "table": table_name,
        "columns": [
            {
                "name": c["name"],
                "type": c.get("type"),
                "pk": c.get("pk", False)
            }
            for c in columns
        ],
        "foreign_keys": foreign_keys
    }

    response = get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    schema_json=json.dumps(schema, indent=2)
                )
            }
        ],
        temperature=0.0,
        max_tokens=120
    )

    return response.choices[0].message.content.strip()


def enrich_tables_with_descriptors(G):
    for node, data in G.nodes(data=True):
        if data.get("tipo") != "tabla":
            continue

        table = data["nombre"]

        columns = []
        for n, d in G.nodes(data=True):
            if d.get("tabla") == table and d.get("tipo") != "tabla":
                columns.append({
                    "name": d["nombre"],
                    "type": d.get("tipo"),
                    "pk": d.get("pk", False)
                })

        foreign_keys = []
        for u, v, edata in G.edges(data=True):
            if edata.get("tipo") == "FK" and u.startswith(table + "."):
                foreign_keys.append({
                    "from": u,
                    "to": v
                })

        descriptor = generate_table_descriptor(
            table_name=table,
            columns=columns,
            foreign_keys=foreign_keys
        )

        G.nodes[node]["descriptor"] = descriptor
        G.nodes[node]["embedding"] = embed_texts(descriptor)

def generate_column_descriptor(
        entity,
        attribute,
        is_primary_key=False,
        is_foreign_key=False,
        references=None
):
    payload = {
        "entity": entity.lower(),
        "attribute": attribute.lower(),
        "primary_key": is_primary_key,
        "foreign_key": is_foreign_key,
        "references": references
    }

    response = get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": COLUMN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": COLUMN_USER_TEMPLATE.format(
                    attribute_json=json.dumps(payload, indent=2)
                )
            }
        ],
        temperature=0.1,
        max_tokens=60
    )

    return response.choices[0].message.content.strip()

def enrich_columns_with_descriptors(G):
    for node, data in G.nodes(data=True):
        # Ignore table nodes
        if "tabla" not in data:
            continue

        entity = data["tabla"].lower()
        attribute = data["nombre"].lower()
        is_pk = data.get("pk", False)

        fk_targets = [
            v for u, v, ed in G.out_edges(node, data=True)
            if ed.get("tipo") == "FK"
        ]

        is_fk = len(fk_targets) > 0
        references = None

        if is_fk:
            ref_table = fk_targets[0].split(".")[0].lower()
            references = ref_table

        descriptor = generate_column_descriptor(
            entity=entity,
            attribute=attribute,
            is_primary_key=is_pk,
            is_foreign_key=is_fk,
            references=references
        )

        G.nodes[node]["descriptor"] = descriptor
        G.nodes[node]["embedding"] = embed_texts(descriptor)


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    parser = argparse.ArgumentParser(description="Data view extraction")
    parser.add_argument("--app", default="cargo",
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app
    G = build_data_graph(f"monoliths/{project}/data.sql")
    enrich_tables_with_descriptors(G)
    enrich_columns_with_descriptors(G)

    print("Nodes:")
    for n, d in G.nodes(data=True):
        print(n, d)

    print("Edges:")
    for u, v, d in G.edges(data=True):
        print(u, "->", v, d)

    export_graph(G, f"monoliths/{project}/vista_datos_{project}.graphml")
