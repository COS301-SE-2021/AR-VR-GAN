import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatToolbarModule } from '@angular/material/toolbar';

import { RouterModule } from '@angular/router';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { UploadsComponent } from './uploads/uploads.component';
import { HeaderComponent } from './header/header.component';
import { CoordsComponent } from './coords/coords.component';
import { LoginComponent } from './login/login.component';
import { LandingComponent } from './landing/landing.component';
import { LoginFormComponent } from './login-form/login-form.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { CookieService } from 'ngx-cookie-service';
import { RegisterComponent } from './register/register.component';
import { RegisterFormComponent } from './register-form/register-form.component';
import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { FormsModule } from '@angular/forms';

@NgModule({
  schemas: [ CUSTOM_ELEMENTS_SCHEMA ],
  declarations: [
    AppComponent,
    UploadsComponent,
    HeaderComponent,
    CoordsComponent,
    LoginComponent,
    LandingComponent,
    LoginFormComponent,
    RegisterComponent,
    RegisterFormComponent
  ],
  imports: [
    FormsModule,
    BrowserModule,
    AppRoutingModule,
    MatGridListModule,
    MatToolbarModule,
    RouterModule.forRoot([
      {path: 'login', component: LoginComponent},
      {path: 'landing', component: LandingComponent},
      {path: 'register', component: RegisterComponent},
      {path: '', redirectTo: '/login', pathMatch: 'full'},
    ]),
    BrowserAnimationsModule,
  ],
  providers: [CookieService],
  bootstrap: [AppComponent]
})
export class AppModule { }
