import modules.flags as flags


def _indent_metadata_lines(text, indent):
    lines = str(text).splitlines() or ['']
    return '\n'.join(f'{indent}{line}' if line else indent.rstrip() for line in lines)


def _format_metadata_scalar(value):
    if value is None:
        return 'None'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _format_metadata_list(values):
    if not values:
        return '  []'

    lines = []
    for index, value in enumerate(values, start=1):
        rendered = _format_metadata_value(value)
        rendered_lines = rendered.splitlines() or ['']
        lines.append(f'  {index}. {rendered_lines[0]}')
        for line in rendered_lines[1:]:
            lines.append(f'     {line}')
    return '\n'.join(lines)


def _format_metadata_mapping(parameters):
    lines = []
    for key, value in parameters.items():
        if isinstance(value, dict):
            lines.append(f'{key}:')
            lines.append(_indent_metadata_lines(_format_metadata_mapping(value), '  '))
        elif isinstance(value, list):
            lines.append(f'{key}:')
            lines.append(_format_metadata_list(value))
        else:
            rendered = _format_metadata_scalar(value)
            if '\n' in rendered:
                lines.append(f'{key}:')
                lines.append(_indent_metadata_lines(rendered, '  '))
            else:
                lines.append(f'{key}: {rendered}')
    return '\n'.join(lines)


def _format_metadata_value(value):
    if isinstance(value, dict):
        return _format_metadata_mapping(value)
    if isinstance(value, list):
        return _format_metadata_list(value)
    return _format_metadata_scalar(value)


def format_metadata_preview(parameters, metadata_scheme):
    lines = []
    if parameters is not None:
        if isinstance(parameters, dict):
            lines.append(_format_metadata_mapping(parameters))
        else:
            lines.append(_format_metadata_scalar(parameters))

    scheme_value = metadata_scheme.value if isinstance(metadata_scheme, flags.MetadataScheme) else None
    has_scheme_field = isinstance(parameters, dict) and parameters.get('metadata_scheme') == scheme_value
    if scheme_value is not None and not has_scheme_field:
        lines.append(f'metadata_scheme: {scheme_value}')

    if not lines:
        return 'No metadata found.'

    return '\n'.join(lines)
