import os
from collections import defaultdict
import networkx as nx
import javalang
import javalang.tree
from mvaa.utils.graph import export_graph
from mvaa.utils.embeddings import  normalize_identifier, embed_texts
from mvaa.utils.groq_client import get_client


class JavaImplementationGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

        self.own_packages = set()
        self.own_classes = set()
        self.own_methods = set()

        self.methods_by_class = defaultdict(set)


    def index_project(self, root_dir):
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

                package = tree.package.name if tree.package else "default"
                self.own_packages.add(package)

                for _, decl in tree:
                    if not isinstance(
                            decl,
                            (javalang.tree.ClassDeclaration,
                             javalang.tree.InterfaceDeclaration)
                    ):
                        continue

                    class_fqn = f"{package}.{decl.name}"
                    self.own_classes.add(class_fqn)

                    for method in decl.methods:
                        self.own_methods.add(f"{class_fqn}.{method.name}")
                        self.methods_by_class[class_fqn].add(method.name)


    def _extract_type_names(self, type_node):
        names = set()

        if not type_node:
            return names

        if hasattr(type_node, "name") and type_node.name:
            names.add(type_node.name)

        if hasattr(type_node, "arguments") and type_node.arguments:
            for arg in type_node.arguments:
                if hasattr(arg, "type"):
                    names |= self._extract_type_names(arg.type)

        return names

    def analyze_file(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = javalang.parse.parse(f.read())
        except Exception:
            return

        package = tree.package.name if tree.package else "default"
        pkg_id = f"package:{package}"

        self.create_node(
            pkg_id,
            nombre=package,
            tipo="paquete",
            descripcion=f"Paquete Java {package}",
            origen={"archivo": filepath}
        )

        for _, decl in tree:
            if not isinstance(
                    decl,
                    (javalang.tree.ClassDeclaration,
                     javalang.tree.InterfaceDeclaration)
            ):
                continue

            self._process_class(package, decl, filepath)

    def _process_class(self, package, class_decl, filepath):
        class_fqn = f"{package}.{class_decl.name}"
        class_id = f"class:{class_fqn}"
        pkg_id = f"package:{package}"

        self.create_node(
            class_id,
            nombre=class_decl.name,
            tipo="clase",
            descripcion=f"Clase {class_decl.name}",
            origen={"archivo": filepath}
        )

        self.add_or_update_edge(
            pkg_id,
            class_id,
            tipo="contiene",
            origen_entry={"archivo": filepath}
        )

        if getattr(class_decl, "extends", None):
            extends_nodes = (
                class_decl.extends
                if isinstance(class_decl.extends, list)
                else [class_decl.extends]
            )
            for ext in extends_nodes:
                for t in self._extract_type_names(ext):
                    self._add_class_dependency(
                        class_fqn,
                        t,
                        "hereda_de",
                        {"archivo": filepath}
                    )

        if isinstance(class_decl, javalang.tree.ClassDeclaration):
            for impl in class_decl.implements or []:
                for t in self._extract_type_names(impl):
                    self._add_class_dependency(
                        class_fqn,
                        t,
                        "implementa",
                        {"archivo": filepath}
                    )

        field_env = {}

        for field in class_decl.fields:
            for decl in field.declarators:
                for t in self._extract_type_names(field.type):
                    field_env[decl.name] = t
                    self._add_class_dependency(
                        class_fqn,
                        t,
                        "usa",
                        {"archivo": filepath, "campo": decl.name}
                    )

        def _is_transactional_class(cls) -> bool:
            anns = self._annotation_names(getattr(cls, "annotations", None))
            # Spring
            if "Transactional" in anns:
                return True
            # Java EE
            if "TransactionAttribute" in anns:
                return True
            # EJBs are transactional by default
            if any(a in anns for a in ["Stateless", "Stateful", "Singleton", "MessageDriven"]):
                return True
            return False

        class_is_transactional = _is_transactional_class(class_decl)
        for method in class_decl.methods:
            self._process_method(
                class_fqn,
                class_id,
                method,
                field_env,
                filepath,
                class_is_transactional=class_is_transactional
            )

        # Constructors
        for ctor in class_decl.constructors:
            self._process_method(
                class_fqn,
                class_id,
                ctor,
                field_env,
                filepath,
                class_is_transactional=class_is_transactional
            )

    def _build_method_type_env(self, method, field_env):
        env = dict(field_env)

        for param in method.parameters:
            for t in self._extract_type_names(param.type):
                env[param.name] = t

        for _, node in method:
            if isinstance(node, javalang.tree.LocalVariableDeclaration):
                for decl in node.declarators:
                    for t in self._extract_type_names(node.type):
                        env[decl.name] = t

        return env

    def _annotation_names(self, ann_list):
        if not ann_list:
            return set()
        names = set()
        for a in ann_list:
            name = getattr(a, "name", None)
            if not name:
                continue
            names.add(str(name).split(".")[-1])
        return names

    def _is_transactional_method(self, method, class_is_transactional: bool = False) -> bool:
        anns = self._annotation_names(getattr(method, "annotations", None))
        if "Transactional" in anns:
            return True
        # Java EE / Jakarta EE
        if "TransactionAttribute" in anns:
            return True
        return bool(class_is_transactional)


    def _process_method(self, class_fqn, class_id, method, field_env, filepath, class_is_transactional=False):
        method_fqn = f"{class_fqn}.{method.name}"
        method_id = f"method:{method_fqn}"

        if isinstance(method, javalang.tree.MethodDeclaration):
            for t in self._extract_type_names(method.return_type):
                self._add_class_dependency(
                    class_fqn, t, "usa",
                    {"archivo": filepath, "metodo": method.name}
                )

        for param in method.parameters:
            for t in self._extract_type_names(param.type):
                self._add_class_dependency(
                    class_fqn, t, "usa",
                    {"archivo": filepath, "metodo": method.name}
                )

        method_ann = self._annotation_names(getattr(method, "annotations", None))
        is_tx = self._is_transactional_method(method, class_is_transactional=class_is_transactional)

        self.create_node(
            method_id,
            nombre=method.name,
            tipo="metodo",
            descripcion=f"Método {method.name}",
            origen={"archivo": filepath},
            annotations=sorted(method_ann),
            transactional=is_tx,
            transactional_source=("method" if "Transactional" in method_ann else ("class" if class_is_transactional else None))
        )

        self.add_or_update_edge(
            class_id,
            method_id,
            tipo="contiene",
            origen_entry={"archivo": filepath}
        )

        if not method.body:
            return

        type_env = self._build_method_type_env(method, field_env)

        for var_name, type_name in type_env.items():
            matches = [
                c for c in self.own_classes
                if c.endswith(f".{type_name}")
            ]

            for target_fqn in matches:
                self._add_class_dependency(
                    class_fqn,
                    type_name,
                    tipo="usa",
                    origen={
                        "archivo": filepath,
                        "metodo": method.name,
                        "variable": var_name,
                    }
                )


        for _, node in method:

            if isinstance(node, javalang.tree.ClassCreator):
                for t in self._extract_type_names(node.type):
                    self._add_class_dependency(
                        class_fqn,
                        t,
                        "usa",
                        {
                            "archivo": filepath,
                            "linea": node.position.line if node.position else None,
                            "metodo": method.name
                        }
                    )

            if not isinstance(node, javalang.tree.MethodInvocation):
                continue

            callee = node.member
            qualifier = node.qualifier
            matches = []

            if qualifier:
                type_name = type_env.get(qualifier)
                if type_name:
                    candidates = [
                        c for c in self.own_classes
                        if c.endswith(f".{type_name}")
                    ]
                else:
                    candidates = [
                        c for c in self.own_classes
                        if c.endswith(f".{qualifier}")
                    ]

                for cls in candidates:
                    if callee in self.methods_by_class.get(cls, []):
                        matches.append(f"{cls}.{callee}")
            else:
                if callee in self.methods_by_class.get(class_fqn, []):
                    matches.append(f"{class_fqn}.{callee}")

            for target in matches:
                target_id = f"method:{target}"
                self.create_node(
                    target_id,
                    nombre=callee,
                    tipo="metodo"
                )
                self.add_or_update_edge(
                    method_id,
                    target_id,
                    tipo="invoca",
                    evidencia="estatico",
                    origen_entry={
                        "archivo": filepath,
                        "linea": node.position.line if node.position else None,
                        "metodo_origen": method.name
                    }
                )

    def _add_class_dependency(self, src_class_fqn, target_simple_name, tipo, origen):
        if not target_simple_name:
            return

        matches = [
            c for c in self.own_classes
            if c.endswith(f".{target_simple_name}")
        ]

        for target_fqn in matches:
            src_id = f"class:{src_class_fqn}"
            tgt_id = f"class:{target_fqn}"

            self.create_node(
                tgt_id,
                nombre=target_simple_name,
                tipo="clase"
            )

            self.add_or_update_edge(
                src_id,
                tgt_id,
                tipo=tipo,
                evidencia="estatico",
                origen_entry=origen
            )

    def create_node(self, node_id, nombre, tipo, descripcion="", origen=None, annotations=[], transactional=False, transactional_source=None):
        if node_id in self.graph:
            return
        self.graph.add_node(
            node_id,
            id=node_id,
            nombre=nombre,
            tipo=tipo,
            descripcion=descripcion,
            metricas={},
            origen=origen or {},
            annotations=annotations,
            transactional=transactional,
            transactional_source=transactional_source
        )

    def add_or_update_edge(self, src, tgt, tipo, origen_entry, evidencia="estatico"):
        for _, data in self.graph.get_edge_data(src, tgt, default={}).items():
            if data.get("tipo") == tipo:
                data["peso"] += 1
                data["origen"].append(origen_entry)
                return

        self.graph.add_edge(
            src,
            tgt,
            tipo=tipo,
            evidencia=evidencia,
            peso=1,
            origen=[origen_entry]
        )


def compute_fan_metrics(graph):
    for node in graph.nodes:
        fan_in = 0
        fan_out = 0

        for _, _, data in graph.in_edges(node, data=True):
            if data.get("tipo") == "invoca":
                fan_in += data.get("peso", 1)

        for _, _, data in graph.out_edges(node, data=True):
            if data.get("tipo") == "invoca":
                fan_out += data.get("peso", 1)

        graph.nodes[node]["metricas"]["fan_in"] = fan_in
        graph.nodes[node]["metricas"]["fan_out"] = fan_out

def extract_class_semantic_features(G, class_id):
    node = G.nodes[class_id]

    assert node.get("tipo") == "clase"

    class_name = normalize_identifier(node.get("nombre", ""))

    package = class_id.split(":")[1].rsplit(".", 1)[0]
    package_name = normalize_identifier(package)

    uses = set()
    inherits = set()
    implements = set()
    methods = set()

    for _, tgt, edata in G.out_edges(class_id, data=True):
        tgt_node = G.nodes[tgt]
        rel = edata.get("tipo")

        if rel == "usa" and tgt_node.get("tipo") == "clase":
            uses.add(normalize_identifier(tgt_node.get("nombre", "")))

        elif rel == "hereda_de":
            inherits.add(normalize_identifier(tgt_node.get("nombre", "")))

        elif rel == "implementa":
            implements.add(normalize_identifier(tgt_node.get("nombre", "")))

        elif rel == "contiene" and tgt_node.get("tipo") == "metodo":
            methods.add(normalize_identifier(tgt_node.get("nombre", "")))

    metrics = node.get("metricas", {})
    fan_in = metrics.get("fan_in", 0)
    fan_out = metrics.get("fan_out", 0)

    return {
        "class_name": class_name,
        "package": package_name,
        "uses": sorted(u for u in uses if u),
        "inherits": sorted(i for i in inherits if i),
        "implements": sorted(i for i in implements if i),
        "methods": sorted(m for m in methods if m),
        "fan_in": fan_in,
        "fan_out": fan_out,
    }

def build_class_descriptor(G, class_id):
    f = extract_class_semantic_features(G, class_id)

    lines = []

    lines.append(
        f"Java class {f['class_name']} located in package {f['package']}."
    )

    if f["inherits"]:
        lines.append(
            "Inherits from " + ", ".join(f["inherits"]) + "."
        )

    if f["implements"]:
        lines.append(
            "Implements interfaces " + ", ".join(f["implements"]) + "."
        )

    if f["uses"]:
        lines.append(
            "Uses related classes " + ", ".join(f["uses"]) + "."
        )

    if f["methods"]:
        lines.append(
            "Defines methods " + ", ".join(f["methods"]) + "."
        )

    if f["fan_in"] or f["fan_out"]:
        lines.append(
            f"Interaction profile: fan in {f['fan_in']}, fan out {f['fan_out']}."
        )

    return " ".join(lines)

def generate_class_descriptor_llm(facts, model="llama-3.1-8b-instant"):
    prompt = f"""
You are given factual information extracted from source code.
Your task is to write a short semantic description of the class responsibility.

Rules:
- Use only the provided information.
- Do NOT assume frameworks, persistence mechanisms, or runtime behavior.
- Do NOT invent responsibilities not directly implied.
- Focus on what the class manages and what operations it provides.
- Keep the description concise (1–2 sentences).

Facts:
- Class name: {facts["class_name"]}
- Package: {facts["package"]}
- Methods: {", ".join(facts["methods"]) if facts["methods"] else "none"}
- Related classes: {", ".join(facts["related_classes"]) if facts["related_classes"] else "none"}

Description:
""".strip()

    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=120,
    )

    return response.choices[0].message.content.strip()


def extract_class_facts(G, class_node):
    data = G.nodes[class_node]

    class_name = data["nombre"]

    package = None
    methods = []
    related_classes = set()

    # package
    for u, v, d in G.in_edges(class_node, data=True):
        if d.get("tipo") == "contiene" and u.startswith("package:"):
            package = G.nodes[u]["nombre"]

    # methods
    for _, v, d in G.out_edges(class_node, data=True):
        if d.get("tipo") == "contiene" and v.startswith("method:"):
            methods.append(G.nodes[v]["nombre"])

    # structural dependencies
    for _, v, d in G.out_edges(class_node, data=True):
        if d.get("tipo") in {"usa", "hereda_de", "implementa"} and v.startswith("class:"):
            related_classes.add(G.nodes[v]["nombre"])

    return {
        "class_name": class_name,
        "package": package or "default",
        "methods": sorted(methods),
        "related_classes": sorted(related_classes),
    }


def enrich_implementation_graph_with_descriptors(G):
    for node_id, data in G.nodes(data=True):
        if data.get("tipo") == "clase":
            descriptor = build_class_descriptor(G, node_id)
            G.nodes[node_id]["semantic_descriptor"] = descriptor
            facts = extract_class_facts(G, node_id)
            llm_descriptor = generate_class_descriptor_llm(facts)
            G.nodes[node_id]["semantic_descriptor_llm"] = llm_descriptor
            G.nodes[node_id]["embedding"] = embed_texts(llm_descriptor)



def build_implementation_graph(root_dir):
    builder = JavaImplementationGraphBuilder()
    builder.index_project(root_dir)

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java"):
                builder.analyze_file(os.path.join(root, file))

    compute_fan_metrics(builder.graph)
    return builder.graph



if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    parser = argparse.ArgumentParser(description="Implementation view extraction")
    parser.add_argument("--app", default="daytrader",
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app
    G = build_implementation_graph(f"monoliths/{project}")
    enrich_implementation_graph_with_descriptors(G)

    print("Nodes:")
    for n, d in G.nodes(data=True):
        print(n, d)

    print("\nEdges:")
    for u, v, d in G.edges(data=True):
        print(u, "->", v, d)

    export_graph(G, f"monoliths/{project}/vista_implementacion_{project}.graphml")

