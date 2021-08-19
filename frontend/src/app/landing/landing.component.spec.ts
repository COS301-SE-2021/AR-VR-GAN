import { HttpClientModule } from '@angular/common/http';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { CoordsComponent } from '../coords/coords.component';
import { HeaderComponent } from '../header/header.component';
import { UploadsComponent } from '../uploads/uploads.component';
import { MatGridListModule } from '@angular/material/grid-list';

import { LandingComponent } from './landing.component';

describe('LandingComponent', () => {
  let component: LandingComponent;
  let fixture: ComponentFixture<LandingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LandingComponent, UploadsComponent, HeaderComponent, CoordsComponent ],
      imports: [ HttpClientModule, RouterTestingModule, MatGridListModule ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(LandingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
