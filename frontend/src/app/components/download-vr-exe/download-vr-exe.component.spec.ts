import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DownloadVrExeComponent } from './download-vr-exe.component';

describe('DownloadVrExeComponent', () => {
  let component: DownloadVrExeComponent;
  let fixture: ComponentFixture<DownloadVrExeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DownloadVrExeComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DownloadVrExeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
