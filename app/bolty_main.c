#include <gtk/gtk.h>

#include "bolty_app.h"

int main(int argc, char **argv)
{
    // Create a new application
    return g_application_run(G_APPLICATION(bolty_app_new()), argc, argv);
}
