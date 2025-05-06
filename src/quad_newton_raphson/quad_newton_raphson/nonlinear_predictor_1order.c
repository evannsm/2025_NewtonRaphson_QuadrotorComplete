#include <stdio.h>
#include <stdlib.h> // Include this for memory allocation
#include <math.h>

// Define a struct for the output vector
typedef struct {
    double x, y, z, vx, vy, vz, roll, pitch, yaw;
} Vector9x1;

// Function to perform the calculations
Vector9x1* performCalculations(double g, double mass, double curr_thrust, double curr_rolldot, double curr_pitchdot, double curr_yawdot,
    double curr_x, double curr_y, double curr_z, double curr_vx, double curr_vy, double curr_vz, double curr_roll, double curr_pitch, double curr_yaw, double integration_step, int integrations_int,
    double thrust_dot, double roll_dd, double pitch_dd, double yaw_dd) {

    // Initialize cumulative changes
    double cumm_change_x = 0.0;
    double cumm_change_y = 0.0;
    double cumm_change_z = 0.0;
    double cumm_change_vx = 0.0;
    double cumm_change_vy = 0.0;
    double cumm_change_vz = 0.0;
    double cumm_change_roll = 0.0;
    double cumm_change_pitch = 0.0;
    double cumm_change_yaw = 0.0;

    double cumm_change_thrust = 0.0;
    double cumm_change_roll_dot = 0.0;
    double cumm_change_pitch_dot = 0.0;
    double cumm_change_yaw_dot = 0.0;

    // Initial conditions
    double xdot = curr_vx;
    double ydot = curr_vy;
    double zdot = curr_vz;

    double sr = sin(curr_roll);
    double sy = sin(curr_yaw);
    double sp = sin(curr_pitch);
    double cr = cos(curr_roll);
    double cp = cos(curr_pitch);
    double cy = cos(curr_yaw);
    double vxdot = -(curr_thrust/mass) * (sr*sy + cr*cy*sp);
    double vydot = -(curr_thrust/mass) * (cr*sy*sp - cy*sr);
    double vzdot = g - (curr_thrust/mass) * (cr*cp);

    double rolldot = curr_rolldot;
    double pitchdot = curr_pitchdot;
    double yawdot = curr_yawdot;

    double roll = curr_roll;
    double pitch = curr_pitch;
    double yaw = curr_yaw;
    double change_vx = 0.0;
    double change_vy = 0.0;
    double change_vz = 0.0;

    for (int i = 0; i < integrations_int; i++) {
        double change_x = (xdot + cumm_change_vx) * integration_step;
        double change_y = (ydot + cumm_change_vy) * integration_step;
        double change_z = (zdot + cumm_change_vz) * integration_step;

        change_vx = vxdot * integration_step;
        change_vy = vydot * integration_step;
        change_vz = vzdot * integration_step;

        double change_roll = (rolldot + cumm_change_roll_dot) * integration_step;
        double change_pitch = (pitchdot + cumm_change_pitch_dot) * integration_step;
        double change_yaw = (yawdot + cumm_change_yaw_dot) * integration_step;

        double change_thrust = thrust_dot * integration_step;
        double change_rolldot = roll_dd * integration_step;
        double change_pitchdot = pitch_dd * integration_step;
        double change_yawdot = yaw_dd * integration_step;

        roll = roll + change_roll;
        pitch = pitch + change_pitch;
        yaw = yaw + change_yaw;
        curr_thrust = curr_thrust + change_thrust;

        sr = sin(roll);
        sy = sin(yaw);
        sp = sin(pitch);
        cr = cos(roll);
        cp = cos(pitch);
        cy = cos(yaw);
        vxdot = -(curr_thrust/mass) * (sr*sy + cr*cy*sp);
        vydot = -(curr_thrust/mass) * (cr*sy*sp - cy*sr);
        vzdot = g - (curr_thrust/mass) * (cr*cp);

        cumm_change_x = cumm_change_x + change_x;
        cumm_change_y = cumm_change_y + change_y;
        cumm_change_z = cumm_change_z + change_z;
        cumm_change_vx = cumm_change_vx + change_vx;
        cumm_change_vy = cumm_change_vy + change_vy;
        cumm_change_vz = cumm_change_vz + change_vz;
        cumm_change_roll = cumm_change_roll + change_roll;
        cumm_change_pitch = cumm_change_pitch + change_pitch;
        cumm_change_yaw = cumm_change_yaw + change_yaw;

        cumm_change_roll_dot = cumm_change_roll_dot + change_rolldot;
        cumm_change_pitch_dot = cumm_change_pitch_dot + change_pitchdot;
        cumm_change_yaw_dot = cumm_change_yaw_dot + change_yawdot;
    }

    double x = curr_x + cumm_change_x;
    double y = curr_y + cumm_change_y;
    double z = curr_z + cumm_change_z;

    double vx = curr_vx + cumm_change_vx;
    double vy = curr_vy + cumm_change_vy;
    double vz = curr_vz + cumm_change_vz;

    roll = curr_roll + cumm_change_roll;
    pitch = curr_pitch + cumm_change_pitch;
    yaw = curr_yaw + cumm_change_yaw;

    // printf("[x,y,z,yaw]: %f, %f, %f, %f\n", x, y, z, yaw);

    // Allocate memory for the output vector
    Vector9x1* result = (Vector9x1*)malloc(sizeof(Vector9x1));

    // Populate the output vector
    result->x = x;
    result->y = y;
    result->z = z;
    result->vx = vx;
    result->vy = vy;
    result->vz = vz;
    result->roll = roll;
    result->pitch = pitch;
    result->yaw = yaw;

    return result; // Return a pointer to the output vector
}
