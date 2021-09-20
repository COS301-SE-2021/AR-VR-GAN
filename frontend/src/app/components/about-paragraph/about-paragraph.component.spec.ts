import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AboutParagraphComponent } from './about-paragraph.component';

describe('AboutParagraphComponent', () => {
  let component: AboutParagraphComponent;
  let fixture: ComponentFixture<AboutParagraphComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AboutParagraphComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(AboutParagraphComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
