/*
 * Copyright (c) 2022 Stephan Vedder <stephan.vedder@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "bolty_image.h"
#include <graphene.h>
#include <gtk/gtk.h>

struct _BoltyImage
{
    GObject parent_instance;

    float *data;
    GdkTexture *texture;
};

static void
bolty_image_snapshot(GdkPaintable *paintable, GdkSnapshot *snapshot, double width, double height)
{
    BoltyImage *image = BOLTY_IMAGE(paintable);

    gtk_snapshot_save(snapshot);

    // Draw the underlying textures
    gtk_snapshot_append_texture(snapshot, image->texture, &GRAPHENE_RECT_INIT(0, 0, width, height));

    gtk_snapshot_restore(snapshot);
}

static GdkPaintableFlags
bolty_image_get_flags(GdkPaintable *paintable)
{
    return GDK_PAINTABLE_STATIC_CONTENTS | GDK_PAINTABLE_STATIC_SIZE;
}

static int
bolty_image_get_intrinsic_width(GdkPaintable *paintable)
{
    BoltyImage *image = BOLTY_IMAGE(paintable);
    return gdk_texture_get_width(image->texture);
}

static int
bolty_image_get_intrinsic_height(GdkPaintable *paintable)
{
    BoltyImage *image = BOLTY_IMAGE(paintable);
    return gdk_texture_get_height(image->texture);
}

static void
bolty_image_paintable_init(GdkPaintableInterface *iface)
{
    iface->snapshot = bolty_image_snapshot;
    iface->get_flags = bolty_image_get_flags;
    iface->get_intrinsic_width = bolty_image_get_intrinsic_width;
    iface->get_intrinsic_height = bolty_image_get_intrinsic_height;
}

/* When defining the GType, we need to implement the GdkPaintable interface */
G_DEFINE_TYPE_WITH_CODE(BoltyImage,
                        bolty_image,
                        G_TYPE_OBJECT,
                        G_IMPLEMENT_INTERFACE(GDK_TYPE_PAINTABLE, bolty_image_paintable_init))

BoltyImage *
bolty_image_new_from_data(size_t w, size_t h, size_t c, float *data)
{
    BoltyImage *image;

    image = g_object_new(BOLTY_TYPE_IMAGE, NULL);

    bolty_image_set_from_data(image, w, h, c, data);

    return image;
}

void bolty_image_set_from_data(BoltyImage *self, size_t w, size_t h, size_t c, float *data)
{
    g_return_if_fail(BOLTY_IS_IMAGE(self));
    GdkMemoryFormat format = GDK_MEMORY_R32G32B32_FLOAT;

    self->texture = gdk_memory_texture_new(w, h, format, g_bytes_new(data, w * h * c * 4), w * c * sizeof(float));
}

static void
bolty_image_class_init(BoltyImageClass *klass)
{
    GObjectClass *object_class = G_OBJECT_CLASS(klass);
}

static void
bolty_image_init(BoltyImage *self)
{
}
