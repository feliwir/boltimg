#pragma once

#include <gtk/gtk.h>

#define BOLTY_APP_TYPE (bolty_app_get_type())
G_DECLARE_FINAL_TYPE(BoltyApp, bolty_app, BOLTY, APP, GtkApplication)

BoltyApp *bolty_app_new(void);
