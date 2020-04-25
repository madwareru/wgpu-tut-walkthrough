#version 450

layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_col;

layout(location=0) out vec4 v_col;

void main() {
    gl_Position = vec4(a_pos, 1.0);
    v_col = vec4(a_col, 1.0);
}