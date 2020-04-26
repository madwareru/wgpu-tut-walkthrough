#version 450

layout(location=0) in vec3 a_pos;
layout(location=1) in vec2 a_tex_coords;

layout(location=0) out vec2 v_tex_coords;

void main() {
    gl_Position = vec4(a_pos, 1.0);
    v_tex_coords = a_tex_coords;
}