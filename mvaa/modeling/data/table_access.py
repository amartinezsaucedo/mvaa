import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pprint import pprint
from collections import defaultdict
import os
import json
import javalang

from mvaa.utils.graph import read_graphml

TABLE_REGEX = re.compile(
    r'\bFROM\s+([A-Z_][A-Z0-9_]*)'
    r'|\bJOIN\s+([A-Z_][A-Z0-9_]*)'
    r'|\bINTO\s+([A-Z_][A-Z0-9_]*)'
    r'|\bUPDATE\s+([A-Z_][A-Z0-9_]*)',
    re.IGNORECASE
)
PRIMITIVE_TYPES = {
    "int", "long", "double", "float", "boolean", "char",
    "Integer", "Long", "Double", "Float", "Boolean", "String", "Void"
}


def camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.upper()


def extract_entity_to_table_jpa(root_dir):
    entity_to_table = {}

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".java"):
                continue

            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
                tree = javalang.parse.parse(src)
            except Exception:
                continue

            package_name = tree.package.name if tree.package else None
            if not package_name:
                # if no package, register without package (unusual in real projects)
                package_name = ""

            for _, decl in tree.filter(javalang.tree.ClassDeclaration):
                annotations = decl.annotations or []
                is_entity = any(
                    a.name == "Entity" or a.name.endswith(".Entity")
                    for a in annotations
                )
                if not is_entity:
                    continue

                entity_simple = decl.name
                table = camel_to_snake(entity_simple)  # default

                for a in annotations:
                    if a.name == "Table" or a.name.endswith(".Table"):
                        # javalang: a.element can be None, list, or dict-like depending on the case
                        elems = a.element or []
                        # in most cases it is a list of ElementValuePair
                        for pair in elems:
                            if getattr(pair, "name", None) == "name":
                                # pair.value is usually a Literal
                                table = camel_to_snake(pair.value.value.strip('"'))
                                break

                fqcn = f"{package_name}.{entity_simple}".strip(".")
                entity_id = f"class:{fqcn}"
                entity_to_table[entity_id] = {f"table:{table.upper()}"}

    return entity_to_table


def _build_simple_to_full(entity_ids):
    m = defaultdict(list)
    for eid in entity_ids:
        simple = str(eid).split(".")[-1]
        m[simple].append(eid)

    simple_to_full = {}
    for simple, cands in m.items():
        if len(cands) == 1:
            simple_to_full[simple] = cands[0]
    return simple_to_full


def extract_jpa_repositories(root_dir, domain_entity_ids):
    mapper_to_entity = {}
    simple_to_full_entity = _build_simple_to_full(domain_entity_ids)

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".java"):
                continue

            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
                tree = javalang.parse.parse(src)
            except Exception:
                continue

            package_name = tree.package.name if tree.package else ""
            package_name = package_name or ""

            for _, decl in tree.filter(javalang.tree.InterfaceDeclaration):
                repo_simple = decl.name

                # minimal repository heuristic
                if not repo_simple.endswith("Repository") and not repo_simple.endswith("RepositoryJPA"):
                    continue

                repo_fqcn = f"{package_name}.{repo_simple}".strip(".")
                repo_id = f"class:{repo_fqcn}"

                # Case 1: CrudRepository<E, ID>
                found = False
                for ext in decl.extends or []:
                    # ext can have a chained sub_type
                    current = ext
                    while current:
                        iface_name = current.name.split(".")[-1]
                        if iface_name == "CrudRepository" and getattr(current, "arguments", None):
                            arg0 = current.arguments[0]
                            if hasattr(arg0, "type") and hasattr(arg0.type, "name"):
                                entity_simple = arg0.type.name
                                entity_id = simple_to_full_entity.get(entity_simple)
                                if entity_id:
                                    mapper_to_entity[repo_id] = entity_id
                                    found = True
                                    break
                        current = current.sub_type
                    if found:
                        break

                if found:
                    continue

                # Case 2: DDD-style repository (search for domain types in signatures)
                candidates = []
                for method in decl.methods or []:
                    if method.return_type and getattr(method.return_type, "name", None):
                        t = method.return_type.name
                        if t in simple_to_full_entity:
                            candidates.append(t)
                    for param in method.parameters or []:
                        t = getattr(param.type, "name", None)
                        if t and t in simple_to_full_entity:
                            candidates.append(t)

                if candidates:
                    freq = {e: candidates.count(e) for e in set(candidates)}
                    entity_simple, count = max(freq.items(), key=lambda x: x[1])
                    if count >= 2 and len(freq) == 1:
                        mapper_to_entity[repo_id] = simple_to_full_entity[entity_simple]

    return mapper_to_entity


def build_mapper_to_tables(mapper_to_entity, entity_to_table):
    mapper_to_tables = {}

    for mapper_id, entity_id in mapper_to_entity.items():
        # mapper_id is now class:...Repo
        mapper_simple = mapper_id.split(".")[-1]
        if not (mapper_simple.endswith("JPA") or mapper_simple.endswith("Hibernate") or mapper_simple.endswith("Repository") or mapper_simple.endswith("RepositoryJPA")):
            continue

        tables = entity_to_table.get(entity_id)
        if tables:
            mapper_to_tables[mapper_id] = set(tables)

    return mapper_to_tables


def extract_all_jpa_mappings(root_dir):
    entity_to_table = extract_entity_to_table_jpa(root_dir)
    domain_entities = set(entity_to_table.keys())  # 'class:pkg.Entity'

    mapper_to_entity = extract_jpa_repositories(root_dir, domain_entities)
    mapper_to_tables = build_mapper_to_tables(mapper_to_entity, entity_to_table)

    return mapper_to_entity, entity_to_table, mapper_to_tables


def extract_repo_to_entity(java_source: str):
    m = re.search(r"CrudRepository<\s*(\w+)\s*,", java_source)
    if m:
        return m.group(1)
    return None


def extract_entity_to_tables(java_source: str):
    tables = set()

    m = re.search(r'@Table\s*\(\s*name\s*=\s*"(\w+)"', java_source)
    if m:
        tables.add(m.group(1))
    else:
        # fallback: naming convention
        m2 = re.search(r'class\s+(\w+)', java_source)
        if m2:
            tables.add(m2.group(1).upper())

    return tables


def extract_tables_from_sql(sql_text):
    tables = set()
    for match in TABLE_REGEX.findall(sql_text):
        for t in match:
            if t:
                tables.add(f"table:{t.upper()}")
    return tables

def extract_myBatis_mappings(xml_path):
    """
    Extracts:
      - mapper -> entity
      - entity -> tables
    from a MyBatis mapper XML file.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None

    namespace = root.attrib.get("namespace")
    if not namespace:
        return None

    mapper_id = f"class:{namespace}"
    mapper_name = namespace.split(".")[-1]

    entity = None
    tables = set()

    for elem in root:
        # resultType / parameterType → entity
        if not entity:
            entity = elem.attrib.get("resultType") or elem.attrib.get("parameterType")
            entity = entity = f"class:org.mybatis.jpetstore.domain.{entity}" if entity else None

        # SQL body
        if elem.text:
            sql = elem.text.strip()
            tables |= extract_tables_from_sql(sql)

    if not entity:
        return None

    return {
        "mapper": mapper_id,
        "entity": entity,
        "tables": tables
    }

def extract_all_mybatis_mappings(root_dir):
    mapper_to_entity = {}
    entity_to_tables = defaultdict(set)
    mapper_to_tables = defaultdict(set)

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".xml"):
                continue

            path = os.path.join(root, file)
            result = extract_myBatis_mappings(path)
            if not result:
                continue

            mapper = result["mapper"]
            entity = result["entity"]
            tables = result["tables"]

            mapper_to_entity[mapper] = entity
            entity_to_tables[entity] |= tables
            mapper_to_tables[mapper] |= tables

    return mapper_to_entity, dict(entity_to_tables), dict(mapper_to_tables)

def iter_interface_chain(ref_type):
    current = ref_type
    while current:
        yield current
        current = current.sub_type


def extract_jpa_repositories_old(root_dir, domain_entities):
    mapper_to_entity = {}

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".java"):
                continue

            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tree = javalang.parse.parse(f.read())
            except Exception:
                continue

            for _, decl in tree.filter(javalang.tree.InterfaceDeclaration):
                repo_name = decl.name
                found = False

                # Minimal "repository" heuristic
                if not repo_name.endswith("Repository") and not repo_name.endswith("RepositoryJPA"):
                    continue

                # Case 1: CrudRepository<E, ID>
                for ext in decl.extends or []:
                    for iface in iter_interface_chain(ext):
                        iface_name = iface.name.split(".")[-1]

                        # Caso 1: CrudRepository<T, ID>
                        if iface_name == "CrudRepository" and iface.arguments:
                            arg0 = iface.arguments[0]
                            if hasattr(arg0, "type"):
                                entity = arg0.type.name
                                mapper_to_entity[repo_name] = entity
                                found = True
                                break
                    if found:
                        break
                else:
                    # Case 2: DDD-style repository
                    candidates = []

                    for method in decl.methods:
                        if method.return_type:
                            t = method.return_type.name
                            if t in domain_entities:
                                candidates.append(t)

                        for param in method.parameters:
                            t = param.type.name
                            if t in domain_entities:
                                candidates.append(t)

                    if candidates:
                        freq = {e: candidates.count(e) for e in set(candidates)}
                        entity, count = max(freq.items(), key=lambda x: x[1])

                        if count >= 2 and len(freq) == 1:
                            mapper_to_entity[repo_name] = entity

    return mapper_to_entity



def get_table_annotation(annotations):
    for a in annotations:
        if a.name == "Table" or a.name.endswith(".Table"):
            return a
    return None

def is_entity_annotation(a):
    return a.name == "Entity" or a.name.endswith(".Entity")


def extract_entity_to_table_jpa_old(root_dir):
    entity_to_table = {}

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".java"):
                continue

            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tree = javalang.parse.parse(f.read())
            except Exception:
                continue

            for package, decl in tree.filter(javalang.tree.ClassDeclaration):
                annotations = decl.annotations

                is_entity = any(
                    a.name == "Entity" or a.name.endswith(".Entity")
                    for a in annotations
                )

                if not is_entity:
                    continue

                entity = decl.name
                table = entity
                package_name = package[0].package.name

                for a in annotations:
                    if a.name == "Table" or a.name.endswith(".Table"):
                        for pair in a.element:
                            if pair.name == "name":
                                table = pair.value.value.strip('"')

                entity_to_table[f"class:{package_name}.{entity}.java"] = {f"table:{table.upper()}"}

    return entity_to_table


def build_mapper_to_tables_old(mapper_to_entity, entity_to_table):
    mapper_to_tables = {}

    for mapper, entity in mapper_to_entity.items():
        # solo infra
        if not (mapper.endswith("JPA") or mapper.endswith("Hibernate")):
            continue

        tables = entity_to_table.get(entity)
        if tables:
            mapper_to_tables[mapper] = set(tables)

    return mapper_to_tables



def extract_all_jpa_mappings_old(root_dir):
    entity_to_table = extract_entity_to_table_jpa(root_dir)
    domain_entities = set(entity_to_table.keys())

    mapper_to_entity = extract_jpa_repositories(root_dir, domain_entities)
    mapper_to_tables = build_mapper_to_tables(mapper_to_entity, entity_to_table)

    return mapper_to_entity, entity_to_table, mapper_to_tables

def extract_ejb_to_tables_jpa(root_dir, entity_to_table, known_tables=None):
    ejb_to_tables = defaultdict(set)
    simple_to_full = _build_simple_to_full(set(entity_to_table.keys()))
    EJB_ANNOTATIONS = {"Stateless", "Stateful", "Singleton", "TransactionAttribute"}

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".java"):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
                tree = javalang.parse.parse(src)
            except Exception:
                continue

            package_name = tree.package.name if tree.package else ""

            for _, decl in tree.filter(javalang.tree.ClassDeclaration):
                annotations = [a.name for a in (decl.annotations or [])]
                is_ejb = any(a in EJB_ANNOTATIONS for a in annotations)
                if not is_ejb:
                    continue

                fqcn   = f"{package_name}.{decl.name}".strip(".")
                cls_id = f"class:{fqcn}"
                tables: set = set()

                for _, type_ref in tree.filter(javalang.tree.ReferenceType):
                    name = getattr(type_ref, "name", None)
                    if name and name in simple_to_full:
                        entity_id = simple_to_full[name]
                        tables |= entity_to_table.get(entity_id, set())

                sql_tables = extract_tables_from_sql(src)
                if known_tables:
                    sql_tables = {t for t in sql_tables
                                  if t.replace("table:", "") in known_tables}
                tables |= sql_tables

                if tables:
                    ejb_to_tables[cls_id] = tables

    return dict(ejb_to_tables)


def propagate_table_access_via_graph(
        mapper_to_tables: dict,
        impl_graphml_path: str,
        known_tables: set = None,
) -> dict:
    G = read_graphml(impl_graphml_path)
    result = dict(mapper_to_tables)

    changed = True
    while changed:
        changed = False
        for u, v, data in G.edges(data=True):
            u, v = str(u), str(v)
            if v in result and u not in result:
                result[u] = set(result[v])
                changed = True
            elif v in result and u in result:
                new_tables = result[u] | result[v]
                if new_tables != result[u]:
                    result[u] = new_tables
                    changed = True

    return result


def extract_jdbc_to_tables(root_dir, known_tables=None):
    jdbc_to_tables = defaultdict(set)
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".java"):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
                tree = javalang.parse.parse(src)
            except Exception:
                continue

            package_name = tree.package.name if tree.package else ""
            for _, decl in tree.filter(javalang.tree.ClassDeclaration):
                fqcn   = f"{package_name}.{decl.name}".strip(".")
                cls_id = f"class:{fqcn}"
                tables = extract_tables_from_sql(src)

                if known_tables:
                    tables = {t for t in tables
                              if t.replace("table:", "") in known_tables}

                if tables:
                    jdbc_to_tables[cls_id] = tables

    return dict(jdbc_to_tables)

def extract_all_daytrader_mappings(root_dir, known_tables=None):
    entity_to_table = extract_entity_to_table_jpa(root_dir)
    ejb_to_tables   = extract_ejb_to_tables_jpa(root_dir, entity_to_table)
    jdbc_to_tables  = extract_jdbc_to_tables(root_dir, known_tables=known_tables)

    mapper_to_tables = {
        cls_id: tables
        for cls_id, tables in {**ejb_to_tables, **jdbc_to_tables}.items()
        if cls_id not in entity_to_table and tables
    }
    return entity_to_table, ejb_to_tables, jdbc_to_tables, mapper_to_tables


def extract_known_tables_from_graphml(data_graphml_path: str) -> set:
    G = read_graphml(data_graphml_path)
    tables = set()
    for n in G.nodes():
        nid = str(n)
        if nid.startswith("table:"):
            tables.add(nid.replace("table:", "").upper())
    return tables


class TableAccessExtractor(ABC):
    """Strategy for extracting entity/table and class/table access mappings from a monolith's source tree."""

    @abstractmethod
    def extract(self, monolith_dir: str, project: str, known_tables: set = None) -> tuple[dict, dict]:
        """Return (entity_to_tables, mapper_to_tables), each NodeId -> Set[NodeId]."""
        raise NotImplementedError


class JpaTableAccessExtractor(TableAccessExtractor):
    """Spring Data JPA: @Entity classes + CrudRepository interfaces (cargo)."""

    def extract(self, monolith_dir, project, known_tables=None):
        _, entity_to_table, mapper_to_tables = extract_all_jpa_mappings(monolith_dir)
        return entity_to_table, mapper_to_tables


class MyBatisTableAccessExtractor(TableAccessExtractor):
    """MyBatis mapper XML files (jpetstore)."""

    def extract(self, monolith_dir, project, known_tables=None):
        _, entity_to_tables, mapper_to_tables = extract_all_mybatis_mappings(monolith_dir)
        return entity_to_tables, mapper_to_tables


class DaytraderTableAccessExtractor(TableAccessExtractor):
    """JPA entities + EJB/JDBC table access, propagated through the implementation graph (daytrader)."""

    def extract(self, monolith_dir, project, known_tables=None):
        entity_to_table, ejb_to_tables, jdbc_to_tables, mapper_to_tables = \
            extract_all_daytrader_mappings(monolith_dir, known_tables=known_tables)

        mapper_to_tables = propagate_table_access_via_graph(
            mapper_to_tables,
            f"{monolith_dir}/vista_implementacion_{project}_c.graphml",
            known_tables=known_tables,
        )

        if known_tables:
            mapper_to_tables = {
                cls_id: {t for t in tables if t.replace("table:", "") in known_tables}
                for cls_id, tables in mapper_to_tables.items()
                if any(t.replace("table:", "") in known_tables for t in tables)
            }

        return entity_to_table, mapper_to_tables


EXTRACTORS: dict[str, TableAccessExtractor] = {
    "cargo": JpaTableAccessExtractor(),
    "jpetstore": MyBatisTableAccessExtractor(),
    "daytrader": DaytraderTableAccessExtractor(),
}


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    parser = argparse.ArgumentParser(description="Extract entity/mapper-to-table access mappings")
    parser.add_argument("--app", default="daytrader",
                        choices=list(EXTRACTORS.keys()),
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app
    root_dir = f"monoliths/{project}"

    known_tables = extract_known_tables_from_graphml(
        f"monoliths/{project}/vista_datos_{project}.graphml"
    )

    extractor = EXTRACTORS[project]
    entity_to_table, mapper_to_tables = extractor.extract(root_dir, project, known_tables=known_tables)

    pprint(mapper_to_tables)

    with open(f"monoliths/{project}/entity_to_tables.json", "w") as f:
        json.dump({k: list(v) for k, v in entity_to_table.items()}, f, indent=2)
    with open(f"monoliths/{project}/mapper_to_tables.json", "w") as f:
        json.dump({k: list(v) for k, v in mapper_to_tables.items()}, f, indent=2)