#version 450


layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_col;

layout(location=0) out vec4 v_color;

void main() {
    v_color = vec4((vec2(1.0, 1.0) + a_pos.xy * 2.0) / 2.0, 0.0, 1.0);
    gl_Position = vec4(a_pos, 1.0);
}