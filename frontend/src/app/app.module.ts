import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatCardModule } from '@angular/material/card';
import { RouterModule } from '@angular/router';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { UploadsComponent } from './uploads/uploads.component';
import { HeaderComponent } from './header/header.component';
import { CoordsComponent } from './coords/coords.component';
import { LoginComponent } from './login/login.component';
import { LandingComponent } from './landing/landing.component';

@NgModule({
  declarations: [
    AppComponent,
    UploadsComponent,
    HeaderComponent,
    CoordsComponent,
    LoginComponent,
    LandingComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    MatGridListModule,
    MatToolbarModule,
    MatFormFieldModule,
    MatCardModule,
    RouterModule.forRoot([
      {path: 'login', component: LoginComponent},
      {path: 'landing', component: LandingComponent},
      {path: '', redirectTo: '/login', pathMatch: 'full'},
    ]),
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
