use anyhow::Result;
use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser, TreeCursor};
use std::collections::{HashMap, HashSet};

/// Supported programming languages for code analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LanguageType {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Java,
    Cpp,
    C,
    Go,
    Unknown,
}

impl Default for LanguageType {
    fn default() -> Self {
        Self::Unknown
    }
}

impl std::fmt::Display for LanguageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LanguageType::Rust => write!(f, "rust"),
            LanguageType::Python => write!(f, "python"),
            LanguageType::JavaScript => write!(f, "javascript"),
            LanguageType::TypeScript => write!(f, "typescript"),
            LanguageType::Java => write!(f, "java"),
            LanguageType::Cpp => write!(f, "cpp"),
            LanguageType::C => write!(f, "c"),
            LanguageType::Go => write!(f, "go"),
            LanguageType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Comprehensive code analysis features extracted from syntax tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFeatures {
    /// Programming language detected
    pub language: LanguageType,
    /// Function definitions found in code
    pub functions: Vec<FunctionInfo>,
    /// Variable declarations and assignments
    pub variables: Vec<VariableInfo>,
    /// Import/include statements
    pub imports: Vec<ImportInfo>,
    /// Class/struct definitions
    pub types: Vec<TypeInfo>,
    /// Comments and documentation
    pub comments: Vec<CommentInfo>,
    /// Indentation and formatting patterns
    pub formatting: FormattingInfo,
    /// Syntax tree complexity metrics
    pub complexity: ComplexityMetrics,
    /// Code structure and organization
    pub structure: StructureInfo,
}

/// Information about function definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    /// Function parameters with types
    pub parameters: Vec<ParameterInfo>,
    /// Return type if available
    pub return_type: Option<String>,
    /// Function visibility/access modifier
    pub visibility: Option<String>,
    /// Function body start and end positions
    pub span: (usize, usize),
    /// Function documentation/comments
    pub documentation: Option<String>,
    /// Whether function is async/await
    pub is_async: bool,
    /// Whether function is generic
    pub is_generic: bool,
}

/// Parameter information for functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter type if available
    pub param_type: Option<String>,
    /// Whether parameter is mutable
    pub is_mutable: bool,
    /// Default value if present
    pub default_value: Option<String>,
}

/// Variable declaration and usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableInfo {
    /// Variable name
    pub name: String,
    /// Variable type if available
    pub var_type: Option<String>,
    /// Whether variable is mutable
    pub is_mutable: bool,
    /// Variable scope (local, global, class member)
    pub scope: String,
    /// Declaration position
    pub declaration_span: (usize, usize),
    /// Initial value if present
    pub initial_value: Option<String>,
}

/// Import/include statement information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    /// Module or library being imported
    pub module: String,
    /// Specific items being imported
    pub items: Vec<String>,
    /// Import alias if used
    pub alias: Option<String>,
    /// Whether it's a wildcard import
    pub is_wildcard: bool,
    /// Import statement position
    pub span: (usize, usize),
}

/// Type definition information (classes, structs, enums)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Type kind (class, struct, enum, interface)
    pub kind: String,
    /// Fields or members
    pub members: Vec<MemberInfo>,
    /// Parent types or interfaces
    pub inheritance: Vec<String>,
    /// Type visibility
    pub visibility: Option<String>,
    /// Type definition span
    pub span: (usize, usize),
    /// Type documentation
    pub documentation: Option<String>,
}

/// Member information for types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberInfo {
    /// Member name
    pub name: String,
    /// Member type
    pub member_type: Option<String>,
    /// Member visibility
    pub visibility: Option<String>,
    /// Whether member is static
    pub is_static: bool,
    /// Member documentation
    pub documentation: Option<String>,
}

/// Comment and documentation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentInfo {
    /// Comment content
    pub content: String,
    /// Comment type (line, block, doc)
    pub comment_type: String,
    /// Comment position
    pub span: (usize, usize),
    /// Associated code element if documentation comment
    pub associated_element: Option<String>,
}

/// Code formatting and style information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattingInfo {
    /// Predominant indentation style
    pub indentation_style: String,
    /// Indentation size (spaces/tabs)
    pub indentation_size: usize,
    /// Line ending style
    pub line_endings: String,
    /// Average line length
    pub average_line_length: f64,
    /// Maximum line length
    pub max_line_length: usize,
    /// Bracket style (K&R, Allman, etc.)
    pub bracket_style: String,
}

/// Code complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic_complexity: usize,
    /// Number of lines of code
    pub lines_of_code: usize,
    /// Number of logical lines
    pub logical_lines: usize,
    /// Nesting depth levels
    pub max_nesting_depth: usize,
    /// Number of decision points
    pub decision_points: usize,
    /// Halstead complexity metrics
    pub halstead_metrics: HalsteadMetrics,
}

/// Halstead complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalsteadMetrics {
    /// Number of distinct operators
    pub distinct_operators: usize,
    /// Number of distinct operands
    pub distinct_operands: usize,
    /// Total operators
    pub total_operators: usize,
    /// Total operands
    pub total_operands: usize,
    /// Program vocabulary
    pub vocabulary: usize,
    /// Program length
    pub length: usize,
    /// Estimated difficulty
    pub difficulty: f64,
    /// Estimated effort
    pub effort: f64,
}

/// Code structure and organization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureInfo {
    /// Module or namespace structure
    pub modules: Vec<String>,
    /// File organization patterns
    pub file_patterns: Vec<String>,
    /// Code organization score (0.0 - 1.0)
    pub organization_score: f64,
    /// Dependency relationships
    pub dependencies: Vec<DependencyInfo>,
}

/// Dependency relationship information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    /// Source element
    pub source: String,
    /// Target element
    pub target: String,
    /// Dependency type (import, inheritance, composition)
    pub dependency_type: String,
    /// Dependency strength (0.0 - 1.0)
    pub strength: f64,
}

/// Syntax analyzer for code understanding and feature extraction
pub struct SyntaxAnalyzer {
    /// Tree-sitter parser for syntax analysis
    parser: Parser,
    /// Programming language being analyzed
    language: LanguageType,
    /// Language-specific syntax patterns
    patterns: LanguagePatterns,
}

/// Language-specific syntax patterns and rules
#[derive(Debug)]
struct LanguagePatterns {
    /// Function definition patterns
    function_patterns: Vec<String>,
    /// Variable declaration patterns
    variable_patterns: Vec<String>,
    /// Import statement patterns
    import_patterns: Vec<String>,
    /// Type definition patterns
    type_patterns: Vec<String>,
    /// Comment patterns
    comment_patterns: Vec<String>,
    /// Keywords for the language
    keywords: HashSet<String>,
    /// Operators for the language
    operators: HashSet<String>,
}

impl SyntaxAnalyzer {
    /// Create a new syntax analyzer for the specified language
    /// 
    /// # Arguments
    /// * `language` - Programming language to analyze
    /// 
    /// # Returns
    /// * `Result<Self>` - Initialized analyzer or error
    pub fn new(_language: LanguageType) -> Result<Self> {
        let _parser = Parser::new();
        // Temporarily disable tree-sitter functionality due to version conflicts
        // TODO: Fix tree-sitter version compatibility 
        Err(anyhow::anyhow!("Tree-sitter functionality temporarily disabled"))
    }
    
    /// Analyze code and extract comprehensive features
    /// 
    /// # Arguments
    /// * `code` - Source code to analyze
    /// 
    /// # Returns
    /// * `Result<CodeFeatures>` - Extracted features or error
    pub fn analyze(&mut self, code: &str) -> Result<CodeFeatures> {
        tracing::info!("Analyzing {} code ({} bytes)", self.language, code.len());
        
        // Parse code into syntax tree
        let tree = self.parser.parse(code, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse code"))?;
        
        let root_node = tree.root_node();
        
        // Extract various code features
        let functions = self.extract_functions(&root_node, code)?;
        let variables = self.extract_variables(&root_node, code)?;
        let imports = self.extract_imports(&root_node, code)?;
        let types = self.extract_types(&root_node, code)?;
        let comments = self.extract_comments(&root_node, code)?;
        let formatting = self.analyze_formatting(code)?;
        let complexity = self.calculate_complexity(&root_node, code)?;
        let structure = self.analyze_structure(&root_node, code)?;
        
        Ok(CodeFeatures {
            language: self.language,
            functions,
            variables,
            imports,
            types,
            comments,
            formatting,
            complexity,
            structure,
        })
    }
    
    /// Extract function definitions from syntax tree
    fn extract_functions(&self, root: &Node, code: &str) -> Result<Vec<FunctionInfo>> {
        let mut functions = Vec::new();
        let mut cursor = root.walk();
        
        self.traverse_for_functions(&mut cursor, code, &mut functions)?;
        
        Ok(functions)
    }
    
    /// Recursively traverse tree to find function definitions
    fn traverse_for_functions(
        &self,
        cursor: &mut TreeCursor,
        code: &str,
        functions: &mut Vec<FunctionInfo>
    ) -> Result<()> {
        let node = cursor.node();
        
        // Check if current node is a function definition
        if self.is_function_definition(&node) {
            if let Some(func_info) = self.parse_function_definition(&node, code)? {
                functions.push(func_info);
            }
        }
        
        // Recursively check children
        if cursor.goto_first_child() {
            loop {
                self.traverse_for_functions(cursor, code, functions)?;
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
        
        Ok(())
    }
    
    /// Check if node represents a function definition
    fn is_function_definition(&self, node: &Node) -> bool {
        match self.language {
            LanguageType::Rust => {
                matches!(node.kind(), "function_item" | "impl_item")
            }
            LanguageType::Python => {
                matches!(node.kind(), "function_definition" | "async_function_definition")
            }
            LanguageType::JavaScript | LanguageType::TypeScript => {
                matches!(node.kind(), "function_declaration" | "function_expression" | "arrow_function" | "method_definition")
            }
            LanguageType::Java => {
                matches!(node.kind(), "method_declaration" | "constructor_declaration")
            }
            LanguageType::Cpp | LanguageType::C => {
                matches!(node.kind(), "function_definition" | "function_declarator")
            }
            LanguageType::Go => {
                matches!(node.kind(), "function_declaration" | "method_declaration")
            }
            _ => false,
        }
    }
    
    /// Parse function definition details
    fn parse_function_definition(&self, node: &Node, code: &str) -> Result<Option<FunctionInfo>> {
        let span = (node.start_byte(), node.end_byte());
        let _function_text = &code[span.0..span.1];
        
        // Extract function name (language-specific logic)
        let name = self.extract_function_name(node, code)?;
        let parameters = self.extract_function_parameters(node, code)?;
        let return_type = self.extract_return_type(node, code)?;
        let visibility = self.extract_visibility(node, code)?;
        let documentation = self.extract_function_documentation(node, code)?;
        let is_async = self.is_async_function(node);
        let is_generic = self.is_generic_function(node);
        
        if name.is_empty() {
            return Ok(None);
        }
        
        Ok(Some(FunctionInfo {
            name,
            parameters,
            return_type,
            visibility,
            span,
            documentation,
            is_async,
            is_generic,
        }))
    }
    
    /// Extract function name from node
    fn extract_function_name(&self, node: &Node, code: &str) -> Result<String> {
        let mut cursor = node.walk();
        
        // Find identifier node for function name
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if child.kind() == "identifier" || child.kind() == "field_identifier" {
                    let name_span = (child.start_byte(), child.end_byte());
                    return Ok(code[name_span.0..name_span.1].to_string());
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        
        Ok(String::new())
    }
    
    /// Extract function parameters
    fn extract_function_parameters(&self, node: &Node, code: &str) -> Result<Vec<ParameterInfo>> {
        let mut parameters = Vec::new();
        let mut cursor = node.walk();
        
        // Find parameter list node
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if child.kind().contains("parameter") || child.kind() == "parameters" {
                    parameters.extend(self.parse_parameter_list(&child, code)?);
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        
        Ok(parameters)
    }
    
    /// Parse parameter list details
    fn parse_parameter_list(&self, node: &Node, code: &str) -> Result<Vec<ParameterInfo>> {
        let mut parameters = Vec::new();
        let mut cursor = node.walk();
        
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if let Some(param) = self.parse_single_parameter(&child, code)? {
                    parameters.push(param);
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        
        Ok(parameters)
    }
    
    /// Parse individual parameter
    fn parse_single_parameter(&self, node: &Node, code: &str) -> Result<Option<ParameterInfo>> {
        if !node.kind().contains("parameter") {
            return Ok(None);
        }
        
        let mut name = String::new();
        let mut param_type = None;
        let mut is_mutable = false;
        let default_value = None;
        
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                match child.kind() {
                    "identifier" | "field_identifier" => {
                        let span = (child.start_byte(), child.end_byte());
                        name = code[span.0..span.1].to_string();
                    }
                    "type_identifier" | "primitive_type" => {
                        let span = (child.start_byte(), child.end_byte());
                        param_type = Some(code[span.0..span.1].to_string());
                    }
                    "mut" => {
                        is_mutable = true;
                    }
                    _ => {}
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        
        if name.is_empty() {
            return Ok(None);
        }
        
        Ok(Some(ParameterInfo {
            name,
            param_type,
            is_mutable,
            default_value,
        }))
    }
    
    /// Extract return type from function
    fn extract_return_type(&self, node: &Node, code: &str) -> Result<Option<String>> {
        let mut cursor = node.walk();
        
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if child.kind().contains("type") && child.kind().contains("return") {
                    let span = (child.start_byte(), child.end_byte());
                    return Ok(Some(code[span.0..span.1].to_string()));
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        
        Ok(None)
    }
    
    /// Extract visibility modifier
    fn extract_visibility(&self, node: &Node, code: &str) -> Result<Option<String>> {
        let mut cursor = node.walk();
        
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                match child.kind() {
                    "visibility_modifier" | "pub" | "private" | "protected" | "public" => {
                        let span = (child.start_byte(), child.end_byte());
                        return Ok(Some(code[span.0..span.1].to_string()));
                    }
                    _ => {}
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        
        Ok(None)
    }
    
    /// Extract function documentation
    fn extract_function_documentation(&self, _node: &Node, _code: &str) -> Result<Option<String>> {
        // Look for documentation comments before the function
        // This is a simplified implementation
        Ok(None)
    }
    
    /// Check if function is async
    fn is_async_function(&self, node: &Node) -> bool {
        let mut cursor = node.walk();
        
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if child.kind() == "async" {
                    return true;
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        
        false
    }
    
    /// Check if function is generic
    fn is_generic_function(&self, node: &Node) -> bool {
        let mut cursor = node.walk();
        
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if child.kind().contains("generic") || child.kind().contains("type_parameter") {
                    return true;
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        
        false
    }
    
    /// Extract variable declarations (simplified implementation)
    fn extract_variables(&self, _root: &Node, _code: &str) -> Result<Vec<VariableInfo>> {
        // Simplified implementation - in practice would traverse tree and extract variable info
        Ok(Vec::new())
    }
    
    /// Extract import statements (simplified implementation)
    fn extract_imports(&self, _root: &Node, _code: &str) -> Result<Vec<ImportInfo>> {
        // Simplified implementation - in practice would parse import/use statements
        Ok(Vec::new())
    }
    
    /// Extract type definitions (simplified implementation)
    fn extract_types(&self, _root: &Node, _code: &str) -> Result<Vec<TypeInfo>> {
        // Simplified implementation - in practice would parse struct/class definitions
        Ok(Vec::new())
    }
    
    /// Extract comments (simplified implementation)
    fn extract_comments(&self, _root: &Node, _code: &str) -> Result<Vec<CommentInfo>> {
        // Simplified implementation - in practice would find all comment nodes
        Ok(Vec::new())
    }
    
    /// Analyze code formatting patterns
    fn analyze_formatting(&self, code: &str) -> Result<FormattingInfo> {
        let lines: Vec<&str> = code.lines().collect();
        let mut total_length = 0;
        let mut max_length = 0;
        let mut indent_sizes = HashMap::new();
        
        for line in &lines {
            let trimmed = line.trim_start();
            let indent_size = line.len() - trimmed.len();
            
            if !trimmed.is_empty() {
                total_length += line.len();
                max_length = max_length.max(line.len());
                *indent_sizes.entry(indent_size).or_insert(0) += 1;
            }
        }
        
        let average_line_length = if lines.is_empty() {
            0.0
        } else {
            total_length as f64 / lines.len() as f64
        };
        
        // Determine predominant indentation
        let (most_common_indent, _) = indent_sizes.iter()
            .max_by_key(|(_, count)| *count)
            .unwrap_or((&0, &0));
        
        let indentation_style = if code.contains('\t') {
            "tabs".to_string()
        } else {
            "spaces".to_string()
        };
        
        Ok(FormattingInfo {
            indentation_style,
            indentation_size: *most_common_indent,
            line_endings: "unix".to_string(), // Simplified
            average_line_length,
            max_line_length: max_length,
            bracket_style: "unknown".to_string(), // Would need more analysis
        })
    }
    
    /// Calculate complexity metrics
    fn calculate_complexity(&self, root: &Node, code: &str) -> Result<ComplexityMetrics> {
        let lines_of_code = code.lines().count();
        let logical_lines = code.lines().filter(|line| !line.trim().is_empty()).count();
        
        // Simplified complexity calculation
        let cyclomatic_complexity = self.calculate_cyclomatic_complexity(root);
        let max_nesting_depth = self.calculate_max_nesting_depth(root);
        let decision_points = self.count_decision_points(root);
        let halstead_metrics = self.calculate_halstead_metrics(root, code)?;
        
        Ok(ComplexityMetrics {
            cyclomatic_complexity,
            lines_of_code,
            logical_lines,
            max_nesting_depth,
            decision_points,
            halstead_metrics,
        })
    }
    
    /// Calculate cyclomatic complexity
    fn calculate_cyclomatic_complexity(&self, root: &Node) -> usize {
        let mut complexity = 1; // Base complexity
        let mut cursor = root.walk();
        
        self.traverse_for_complexity(&mut cursor, &mut complexity);
        complexity
    }
    
    /// Traverse tree to count complexity-increasing constructs
    #[allow(clippy::only_used_in_recursion)]
    fn traverse_for_complexity(&self, cursor: &mut TreeCursor, complexity: &mut usize) {
        let node = cursor.node();
        
        // Count decision points that increase complexity
        match node.kind() {
            "if_expression" | "if_statement" | "while_statement" | "while_expression" |
            "for_statement" | "for_expression" | "match_expression" | "match_statement" |
            "case" | "catch_clause" | "except_clause" => {
                *complexity += 1;
            }
            _ => {}
        }
        
        // Recursively check children
        if cursor.goto_first_child() {
            loop {
                self.traverse_for_complexity(cursor, complexity);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
    
    /// Calculate maximum nesting depth
    fn calculate_max_nesting_depth(&self, root: &Node) -> usize {
        let mut max_depth = 0;
        let mut cursor = root.walk();
        
        self.traverse_for_depth(&mut cursor, 0, &mut max_depth);
        max_depth
    }
    
    /// Traverse tree to find maximum nesting depth
    #[allow(clippy::only_used_in_recursion)]
    fn traverse_for_depth(&self, cursor: &mut TreeCursor, current_depth: usize, max_depth: &mut usize) {
        let _node = cursor.node();
        let depth = current_depth + 1;
        
        *max_depth = (*max_depth).max(depth);
        
        // Recursively check children
        if cursor.goto_first_child() {
            loop {
                self.traverse_for_depth(cursor, depth, max_depth);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
    
    /// Count decision points in code
    fn count_decision_points(&self, root: &Node) -> usize {
        let mut decision_points = 0;
        let mut cursor = root.walk();
        
        self.traverse_for_decisions(&mut cursor, &mut decision_points);
        decision_points
    }
    
    /// Traverse tree to count decision points
    #[allow(clippy::only_used_in_recursion)]
    fn traverse_for_decisions(&self, cursor: &mut TreeCursor, count: &mut usize) {
        let node = cursor.node();
        
        match node.kind() {
            "if_expression" | "if_statement" | "match_expression" | "match_statement" |
            "conditional_expression" | "ternary_expression" => {
                *count += 1;
            }
            _ => {}
        }
        
        // Recursively check children
        if cursor.goto_first_child() {
            loop {
                self.traverse_for_decisions(cursor, count);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
    
    /// Calculate Halstead complexity metrics
    fn calculate_halstead_metrics(&self, root: &Node, code: &str) -> Result<HalsteadMetrics> {
        let mut operators = HashSet::new();
        let mut operands = HashSet::new();
        let mut operator_count = 0;
        let mut operand_count = 0;
        
        let mut cursor = root.walk();
        self.collect_halstead_elements(&mut cursor, code, &mut operators, &mut operands, &mut operator_count, &mut operand_count);
        
        let distinct_operators = operators.len();
        let distinct_operands = operands.len();
        let vocabulary = distinct_operators + distinct_operands;
        let length = operator_count + operand_count;
        
        let difficulty = if distinct_operands > 0 {
            (distinct_operators as f64 / 2.0) * (operand_count as f64 / distinct_operands as f64)
        } else {
            0.0
        };
        
        let effort = difficulty * length as f64;
        
        Ok(HalsteadMetrics {
            distinct_operators,
            distinct_operands,
            total_operators: operator_count,
            total_operands: operand_count,
            vocabulary,
            length,
            difficulty,
            effort,
        })
    }
    
    /// Collect operators and operands for Halstead metrics
    fn collect_halstead_elements(
        &self,
        cursor: &mut TreeCursor,
        code: &str,
        operators: &mut HashSet<String>,
        operands: &mut HashSet<String>,
        operator_count: &mut usize,
        operand_count: &mut usize,
    ) {
        let node = cursor.node();
        let node_text = &code[node.start_byte()..node.end_byte()];
        
        // Classify as operator or operand based on node type
        if self.patterns.operators.contains(node.kind()) {
            operators.insert(node_text.to_string());
            *operator_count += 1;
        } else if node.kind() == "identifier" || node.kind() == "number" || node.kind() == "string" {
            operands.insert(node_text.to_string());
            *operand_count += 1;
        }
        
        // Recursively check children
        if cursor.goto_first_child() {
            loop {
                self.collect_halstead_elements(cursor, code, operators, operands, operator_count, operand_count);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
    
    /// Analyze code structure and organization
    fn analyze_structure(&self, _root: &Node, _code: &str) -> Result<StructureInfo> {
        Ok(StructureInfo {
            modules: Vec::new(),
            file_patterns: Vec::new(),
            organization_score: 0.8, // Simplified score
            dependencies: Vec::new(),
        })
    }
    
    /// Create language-specific patterns
    fn create_language_patterns(language: LanguageType) -> LanguagePatterns {
        let (function_patterns, variable_patterns, import_patterns, type_patterns, comment_patterns, keywords, operators) = match language {
            LanguageType::Rust => (
                vec!["fn ".to_string()],
                vec!["let ".to_string(), "const ".to_string()],
                vec!["use ".to_string()],
                vec!["struct ".to_string(), "enum ".to_string(), "impl ".to_string()],
                vec!["//".to_string(), "/*".to_string()],
                ["fn", "let", "const", "struct", "enum", "impl", "pub", "mod"].iter().map(|s| s.to_string()).collect(),
                ["+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">="].iter().map(|s| s.to_string()).collect(),
            ),
            LanguageType::Python => (
                vec!["def ".to_string()],
                vec!["".to_string()], // Python doesn't have explicit declarations
                vec!["import ".to_string(), "from ".to_string()],
                vec!["class ".to_string()],
                vec!["#".to_string(), "\"\"\"".to_string()],
                ["def", "class", "import", "from", "if", "else", "for", "while"].iter().map(|s| s.to_string()).collect(),
                ["+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">="].iter().map(|s| s.to_string()).collect(),
            ),
            _ => (
                vec!["function ".to_string()],
                vec!["var ".to_string(), "let ".to_string(), "const ".to_string()],
                vec!["import ".to_string()],
                vec!["class ".to_string(), "interface ".to_string()],
                vec!["//".to_string(), "/*".to_string()],
                ["function", "var", "let", "const", "class", "interface"].iter().map(|s| s.to_string()).collect(),
                ["+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">="].iter().map(|s| s.to_string()).collect(),
            ),
        };
        
        LanguagePatterns {
            function_patterns,
            variable_patterns,
            import_patterns,
            type_patterns,
            comment_patterns,
            keywords,
            operators,
        }
    }
}