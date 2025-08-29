"""
Dependency graph builder that uses AST to understand code relationships.
Tracks imports, function calls, and class relationships across the codebase.
"""

import ast
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque


class DependencyNode:
    """Represents a node in the dependency graph."""
    
    def __init__(self, name: str, node_type: str, file_path: str, line_number: int = 0):
        self.name = name
        self.node_type = node_type  # "function", "class", "module", "variable"
        self.file_path = file_path
        self.line_number = line_number
        self.dependencies: Set[str] = set()  # What this node depends on
        self.dependents: Set[str] = set()    # What depends on this node


class DependencyGraphBuilder:
    """Builds and maintains a dependency graph of a codebase."""
    
    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}
        self.imports: Dict[str, Set[str]] = defaultdict(set)  # file -> set of imported modules
        self.exports: Dict[str, Set[str]] = defaultdict(set)  # file -> set of exported symbols
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)  # caller -> set of callees
        
    def _get_node_id(self, name: str, file_path: str) -> str:
        """Generate unique node ID."""
        return f"{file_path}::{name}"
    
    def _analyze_python_file(self, file_path: str) -> Tuple[Set[str], Set[str], Dict[str, Set[str]]]:
        """
        Analyze a Python file to extract imports, exports, and function calls.
        
        Returns:
            Tuple of (imports, exports, function_calls)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError):
            return set(), set(), {}
            
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return set(), set(), {}
        
        imports = set()
        exports = set()
        function_calls = defaultdict(set)
        current_scope = []
        
        # Store discovered nodes for later creation
        discovered_nodes = []
        
        class AnalysisVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.add(alias.name)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    for alias in node.names:
                        if alias.name == '*':
                            imports.add(f"{node.module}.*")
                        else:
                            imports.add(f"{node.module}.{alias.name}")
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                func_name = node.name
                exports.add(func_name)
                
                # Store node info for later creation
                discovered_nodes.append({
                    'name': func_name,
                    'type': 'function',
                    'line_number': node.lineno
                })
                
                # Enter function scope
                current_scope.append(func_name)
                
                # Visit function body to find calls
                self.generic_visit(node)
                
                # Exit function scope
                current_scope.pop()
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)  # Same logic as regular function
            
            def visit_ClassDef(self, node):
                class_name = node.name
                exports.add(class_name)
                
                # Store node info for later creation
                discovered_nodes.append({
                    'name': class_name,
                    'type': 'class',
                    'line_number': node.lineno
                })
                
                # Enter class scope
                current_scope.append(class_name)
                self.generic_visit(node)
                current_scope.pop()
            
            def visit_Call(self, node):
                # Extract function/method calls
                caller = '.'.join(current_scope) if current_scope else "__module__"
                
                if isinstance(node.func, ast.Name):
                    # Simple function call: foo()
                    callee = node.func.id
                    function_calls[caller].add(callee)
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.method() or module.function()
                    if isinstance(node.func.value, ast.Name):
                        callee = f"{node.func.value.id}.{node.func.attr}"
                        function_calls[caller].add(callee)
                    else:
                        # More complex attribute access
                        callee = node.func.attr
                        function_calls[caller].add(callee)
                
                self.generic_visit(node)
        
        visitor = AnalysisVisitor()
        visitor.visit(tree)
        
        # Create nodes after analysis is complete
        for node_info in discovered_nodes:
            node_id = self._get_node_id(node_info['name'], file_path)
            self.nodes[node_id] = DependencyNode(
                name=node_info['name'],
                node_type=node_info['type'],
                file_path=file_path,
                line_number=node_info['line_number']
            )
        
        return imports, exports, function_calls
    
    def build_graph(self, repo_path: str) -> Dict[str, DependencyNode]:
        """
        Build dependency graph for the entire repository.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            Dictionary of node_id -> DependencyNode
        """
        repo_path = Path(repo_path)
        
        # First pass: collect all symbols and basic dependencies
        for file_path in repo_path.rglob('*.py'):
            if file_path.is_file():
                file_str = str(file_path)
                imports, exports, function_calls = self._analyze_python_file(file_str)
                
                self.imports[file_str] = imports
                self.exports[file_str] = exports
                
                # Add function call relationships
                for caller, callees in function_calls.items():
                    caller_id = self._get_node_id(caller, file_str)
                    self.call_graph[caller_id].update(callees)
        
        # Second pass: resolve dependencies
        self._resolve_dependencies()
        
        print(f"Built dependency graph with {len(self.nodes)} nodes")
        return self.nodes
    
    def _resolve_dependencies(self):
        """Resolve string references to actual node dependencies."""
        
        # Create lookup maps for faster resolution
        name_to_nodes = defaultdict(list)
        for node_id, node in self.nodes.items():
            name_to_nodes[node.name].append(node_id)
        
        # Resolve function calls
        for caller_id, callees in self.call_graph.items():
            if caller_id in self.nodes:
                for callee_name in callees:
                    # Try to find the callee in the same file first
                    caller_file = self.nodes[caller_id].file_path
                    
                    # Look for exact matches in the same file
                    same_file_matches = [
                        nid for nid in name_to_nodes.get(callee_name, [])
                        if self.nodes[nid].file_path == caller_file
                    ]
                    
                    if same_file_matches:
                        for match in same_file_matches:
                            self.nodes[caller_id].dependencies.add(match)
                            self.nodes[match].dependents.add(caller_id)
                    else:
                        # Look for matches in imported modules
                        for possible_match in name_to_nodes.get(callee_name, []):
                            self.nodes[caller_id].dependencies.add(possible_match)
                            self.nodes[possible_match].dependents.add(caller_id)
    
    def get_dependencies(self, node_name: str, file_path: str = None, 
                        max_depth: int = 2) -> Set[str]:
        """
        Get all dependencies of a node up to max_depth.
        
        Args:
            node_name: Name of the node to analyze
            file_path: Optional file path to disambiguate
            max_depth: Maximum depth to traverse
            
        Returns:
            Set of node IDs that the target depends on
        """
        # Find the target node
        target_nodes = []
        if file_path:
            node_id = self._get_node_id(node_name, file_path)
            if node_id in self.nodes:
                target_nodes = [node_id]
        else:
            # Search all nodes with matching name
            target_nodes = [
                nid for nid, node in self.nodes.items()
                if node.name == node_name
            ]
        
        if not target_nodes:
            return set()
        
        # BFS to find dependencies
        dependencies = set()
        queue = deque([(nid, 0) for nid in target_nodes])
        visited = set()
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            dependencies.add(current_id)
            
            # Add direct dependencies to queue
            if current_id in self.nodes:
                for dep_id in self.nodes[current_id].dependencies:
                    if dep_id not in visited:
                        queue.append((dep_id, depth + 1))
        
        return dependencies
    
    def get_dependents(self, node_name: str, file_path: str = None,
                      max_depth: int = 2) -> Set[str]:
        """
        Get all dependents of a node up to max_depth.
        
        Args:
            node_name: Name of the node to analyze  
            file_path: Optional file path to disambiguate
            max_depth: Maximum depth to traverse
            
        Returns:
            Set of node IDs that depend on the target
        """
        # Find the target node
        target_nodes = []
        if file_path:
            node_id = self._get_node_id(node_name, file_path)
            if node_id in self.nodes:
                target_nodes = [node_id]
        else:
            # Search all nodes with matching name
            target_nodes = [
                nid for nid, node in self.nodes.items()
                if node.name == node_name
            ]
        
        if not target_nodes:
            return set()
        
        # BFS to find dependents
        dependents = set()
        queue = deque([(nid, 0) for nid in target_nodes])
        visited = set()
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            dependents.add(current_id)
            
            # Add direct dependents to queue
            if current_id in self.nodes:
                for dep_id in self.nodes[current_id].dependents:
                    if dep_id not in visited:
                        queue.append((dep_id, depth + 1))
        
        return dependents
    
    def get_related_context(self, node_name: str, file_path: str = None) -> Set[str]:
        """
        Get comprehensive context around a node (dependencies + dependents).
        
        Args:
            node_name: Name of the node to analyze
            file_path: Optional file path to disambiguate
            
        Returns:
            Set of node IDs representing full context
        """
        dependencies = self.get_dependencies(node_name, file_path)
        dependents = self.get_dependents(node_name, file_path)
        
        return dependencies.union(dependents)
    
    def find_nodes_by_name(self, name: str) -> List[DependencyNode]:
        """Find all nodes with a given name."""
        return [node for node in self.nodes.values() if node.name == name]
    
    def find_nodes_in_file(self, file_path: str) -> List[DependencyNode]:
        """Find all nodes in a specific file."""
        return [node for node in self.nodes.values() if node.file_path == file_path]
    
    def get_file_imports(self, file_path: str) -> Set[str]:
        """Get all imports for a specific file."""
        return self.imports.get(file_path, set())
    
    def get_file_exports(self, file_path: str) -> Set[str]:
        """Get all exports from a specific file.""" 
        return self.exports.get(file_path, set())


if __name__ == "__main__":
    # Example usage
    builder = DependencyGraphBuilder()
    
    # Build graph for a repository
    repo_path = "./test_repo"  # Replace with actual repo path
    nodes = builder.build_graph(repo_path)
    
    # Example queries
    if nodes:
        # Find a function's dependencies
        sample_node = list(nodes.values())[0]
        deps = builder.get_dependencies(sample_node.name, sample_node.file_path)
        print(f"Dependencies of {sample_node.name}: {len(deps)} items")
        
        # Find what depends on a function
        dependents = builder.get_dependents(sample_node.name, sample_node.file_path)
        print(f"Dependents of {sample_node.name}: {len(dependents)} items")
