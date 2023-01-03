#include <gtk/gtk.h>

#include "boltimg.h"
#include "bolty_image.h"

#define STBI_MALLOC bolt_alloc
#define STBI_FREE bolt_free
#define STBI_REALLOC bolt_realloc

#define STB_IMAGE_IMPLEMENTATION
#include "../deps/stb_image.h"

static void on_activate(GtkApplication *app)
{
    int w = 0, h = 0, c = 0;
    stbi_uc *rgb8 = stbi_load("/home/stephan/Development/boltimg/test/assets/lena.png", &w, &h, &c, 3);
    BoltContext ctx;
    bolt_ctx_init(&ctx, BOLT_HL_AUTO);
    float *rgbf32 = bolt_alloc(sizeof(float) * w * h * c);
    bolt_conv_u8_f32_norm(&ctx, w, h, c, rgb8, rgbf32);

    BoltyImage* img = bolty_image_new_from_data(w, h, c, rgbf32);

    // Create a new window
    GtkWidget *window = gtk_application_window_new(app);
    // Create a new button
    GtkWidget *picture = gtk_picture_new_for_paintable(img);
    gtk_window_set_child(GTK_WINDOW(window), picture);
    gtk_window_present(GTK_WINDOW(window));
}

int main(int argc, char **argv)
{
    // Create a new application
    GtkApplication *app = gtk_application_new("org.feliwir.Bolty",
                                              G_APPLICATION_FLAGS_NONE);
    g_signal_connect(app, "activate", G_CALLBACK(on_activate), NULL);
    return g_application_run(G_APPLICATION(app), argc, argv);
}
