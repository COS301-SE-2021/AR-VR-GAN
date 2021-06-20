import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatToolbarModule } from '@angular/material/toolbar';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { UploadsComponent } from './uploads/uploads.component';
import { HeaderComponent } from './header/header.component';
import { CoordsComponent } from './coords/coords.component';
import { ProtoComponent } from './proto/proto.component';

@NgModule({
  declarations: [
    AppComponent,
    UploadsComponent,
    HeaderComponent,
    CoordsComponent,
    ProtoComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    MatGridListModule,
    MatToolbarModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
