# Structured Tool Parameter Validation

The `validate_tool_parameters()` method has been enhanced to provide structured, field-level validation errors instead of concatenated error strings.

## Before (Concatenated Errors)
```python
# Old error format - single concatenated string
try:
    await manager.validate_tool_parameters(tool_name, params)
except McpError as e:
    # e.error.message: "Parameter validation failed for tool 'my_tool': name: Field required; age: Input should be a valid integer"
    print(f"Validation failed: {e.error.message}")
```

## After (Structured Errors)
```python
# New error format - structured field-level errors
try:
    await manager.validate_tool_parameters(tool_name, params)
except McpError as e:
    # Access structured error data
    if e.error.data and "validation_errors" in e.error.data:
        validation_errors = e.error.data["validation_errors"]
        
        for error in validation_errors:
            field = error["field"]          # e.g., "name", "settings.theme"
            message = error["message"]      # e.g., "Field required"
            error_type = error["type"]      # e.g., "missing", "string_type"
            input_value = error.get("input")  # The invalid input value
            
            print(f"Field '{field}': {message}")
            if input_value is not None:
                print(f"  Invalid input: {input_value}")
```

## Error Structure
Each validation error now contains:
- `field`: Dot-notation path to the field (e.g., "settings.notifications")
- `message`: Human-readable error description
- `type`: Pydantic error type (e.g., "missing", "string_type", "bool_parsing")
- `input`: The actual input value that caused the error (when available)

## Additional Data
The `McpError.error.data` also includes:
- `tool_name`: Name of the tool being validated
- `validation_errors`: List of structured field errors
- `parameters_received`: The original parameters that failed validation

## UI Integration Example
```python
def display_validation_errors(mcp_error):
    """Display user-friendly validation errors in a UI."""
    if not mcp_error.error.data or "validation_errors" not in mcp_error.error.data:
        # Fallback to summary message
        return mcp_error.error.message
    
    validation_errors = mcp_error.error.data["validation_errors"]
    error_list = []
    
    for error in validation_errors:
        field_display = error["field"].replace(".", " → ")  # "settings.theme" → "settings → theme"
        error_list.append(f"• {field_display}: {error['message']}")
    
    return "Validation errors:\n" + "\n".join(error_list)
```

This enhancement enables precise, actionable feedback for users and better integration with UI frameworks that can highlight specific form fields with validation errors.
