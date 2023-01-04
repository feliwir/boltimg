/*
 * Copyright 2023 Stephan Vedder
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "bolty_window.h"
#include "bolty_image.h"

#include "boltimg.h"

#define STBI_MALLOC bolt_alloc
#define STBI_FREE bolt_free
#define STBI_REALLOC bolt_realloc

#define STB_IMAGE_IMPLEMENTATION
#include "../deps/stb_image.h"

struct _BoltyWindow
{
    GtkApplicationWindow parent;

    GtkPicture *picture;

    // Bolt stuff
    BoltContext bolt_ctx;
    float *img_data;
};

G_DEFINE_TYPE(BoltyWindow, bolty_window, GTK_TYPE_APPLICATION_WINDOW)

static void
bolty_window_init(BoltyWindow *win)
{
    // Initialize boltimg
    bolt_ctx_init(&win->bolt_ctx, BOLT_HL_AUTO);

    gtk_widget_init_template(GTK_WIDGET(win));
}

static void
bolty_window_class_init(BoltyWindowClass *class)
{
    gtk_widget_class_set_template_from_resource(GTK_WIDGET_CLASS(class),
                                                "/org/feliwir/bolty/bolty-window.ui");
    gtk_widget_class_bind_template_child(GTK_WIDGET_CLASS(class), BoltyWindow, picture);
}

BoltyWindow *
bolty_window_new(BoltyApp *app)
{
    return g_object_new(BOLTY_WINDOW_TYPE, "application", app, NULL);
}

static void
bolty_window_file_read_cb(GObject *source, GAsyncResult *result, gpointer user_data)
{
    g_autoptr(GBytes) bytes = NULL;
    g_autoptr(GError) error = NULL;
    gchar *etag = NULL;
    BoltyWindow *win = BOLTY_WINDOW(user_data);

    bytes = g_file_load_bytes_finish(G_FILE(source), result, &etag, &error);

    // Load the actual image
    int w = 0, h = 0, c = 0;
    stbi_uc *rgb8 = stbi_load_from_memory(g_bytes_get_data(bytes, NULL), g_bytes_get_size(bytes), &w, &h, &c, 3);
    win->img_data = bolt_alloc(sizeof(float) * w * h * c);
    bolt_conv_u8_f32_norm(&win->bolt_ctx, w, h, c, rgb8, win->img_data);
    BoltyImage *img = bolty_image_new_from_data(w, h, c, win->img_data);
    stbi_image_free(rgb8);

    gtk_picture_set_paintable(win->picture, GDK_PAINTABLE(img));
}

void bolty_window_open(BoltyWindow *win,
                       GFile *file)
{
    g_file_load_bytes_async(file, NULL, bolty_window_file_read_cb, win);
}