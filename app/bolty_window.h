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

#pragma once

#include <gtk/gtk.h>
#include "bolty_app.h"

G_BEGIN_DECLS

#define BOLTY_WINDOW_TYPE (bolty_window_get_type())

G_DECLARE_FINAL_TYPE(BoltyWindow, bolty_window, BOLTY, WINDOW, GtkApplicationWindow)

BoltyWindow *bolty_window_new(BoltyApp *app);
void bolty_window_open(BoltyWindow *win,
                       GFile *file);

G_END_DECLS
