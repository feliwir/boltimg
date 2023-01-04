#include <gtk/gtk.h>

#include "bolty_app.h"
#include "bolty_window.h"

struct _BoltyApp
{
    GtkApplication parent;
};

G_DEFINE_TYPE(BoltyApp, bolty_app, GTK_TYPE_APPLICATION);

static void
bolty_app_init(BoltyApp *app)
{
}

static void
bolty_app_activate(GApplication *app)
{
    BoltyWindow *win;

    win = bolty_window_new(BOLTY_APP(app));
    gtk_window_present(GTK_WINDOW(win));
}

static void
bolty_app_open(GApplication *app,
               GFile **files,
               int n_files,
               const char *hint)
{
    GList *windows;
    BoltyWindow *win;
    int i;

    windows = gtk_application_get_windows(GTK_APPLICATION(app));
    if (windows)
        win = BOLTY_WINDOW(windows->data);
    else
        win = bolty_window_new(BOLTY_APP(app));

    for (i = 0; i < n_files; i++)
        bolty_window_open(win, files[i]);

    gtk_window_present(GTK_WINDOW(win));
}

static void
bolty_app_class_init(BoltyAppClass *class)
{
    G_APPLICATION_CLASS(class)->activate = bolty_app_activate;
    G_APPLICATION_CLASS(class)->open = bolty_app_open;
}

BoltyApp *
bolty_app_new(void)
{
    return g_object_new(BOLTY_APP_TYPE,
                        "application-id", "org.feliwir.bolty",
                        "flags", G_APPLICATION_HANDLES_OPEN,
                        NULL);
}
